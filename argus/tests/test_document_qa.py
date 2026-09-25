"""`research/document_qa.py` and `eval/document_qa_eval.py`, with scripted models only.

No test here reaches Qwen or EDGAR: the model is a script and the documents are either written
inline or read from the frozen corpus under ``data/document_qa_corpus/``.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from argus.eval import document_qa_eval as ev
from argus.llm.qwen import Completion, Usage
from argus.lui.answer import Source
from argus.research import document_qa as dq


def _doc(text: str, doc_id: str = "NVDA-10-Q-x", form: str = "10-Q") -> dq.Document:
    return dq.Document(doc_id=doc_id, ticker="NVDA", form=form, filed=date(2026, 8, 26),
                       url=f"https://www.sec.gov/{doc_id}.htm", title=f"NVDA {form}", text=text)


class Scripted:
    """A `ChatModel` whose reply is computed from the prompt it receives."""

    budget: Any = None

    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.prompts: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        self.prompts.append(messages)
        text = self._reply(messages) if callable(self._reply) else self._reply
        return Completion(content=text, reasoning="", finish_reason="stop",
                          usage=Usage(10, 10, 0, 20))

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


FILING = "\n".join([
    "Revenue | $ | 96,221 | $ | 46,743",
    "Data Center revenue was $89.0 billion, up 117% from a year ago, driven by Blackwell Ultra.",
    "Gross margin increased to 75.0% for the second quarter of fiscal year 2027.",
    "We had approximately 42,000 employees in 38 countries.",
] * 1)


# --- text and chunking ------------------------------------------------------------------------


def test_html_to_text_keeps_rows_and_drops_machine_header() -> None:
    raw = ("<html><head><title>x</title></head><body><ix:header>SECRETFACTS 123</ix:header>"
           "<script>var a=1;</script><p>Revenue grew.</p><table><tr><td>Revenue</td><td>$</td>"
           "<td>96,221</td></tr><tr><td>Cost</td><td>10</td></tr></table>"
           "<div style=\"display:none\">hidden</div></body></html>")
    text = dq.html_to_text(raw)
    assert "SECRETFACTS" not in text and "var a" not in text and "hidden" not in text
    lines = text.split("\n")
    assert "Revenue grew." in lines
    assert "Revenue | $ | 96,221" in lines
    assert "Cost | 10" in lines


def test_chunk_ids_are_stable_unique_and_hash_shaped() -> None:
    long = "\n".join(f"Line {i} of the filing says something about revenue {i * 7}." for i in
                     range(300))
    a, b = dq.chunk(_doc(long)), dq.chunk(_doc(long))
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a) > 5
    assert all(re.fullmatch(r"dqa-[0-9a-f]{8}", c.id) for c in a)
    assert all(len(c.text) <= dq.CHUNK_CHARS for c in a)
    # Overlap: the last line of a chunk opens the next one.
    assert a[1].text.split("\n")[0] in a[0].text


def test_chunk_ids_differ_across_documents_with_the_same_text() -> None:
    a = dq.chunk(_doc("Same text.", doc_id="A"))
    b = dq.chunk(_doc("Same text.", doc_id="B"))
    assert a[0].id != b[0].id


def test_duplicate_openings_never_collide() -> None:
    text = "\n".join(["x" * 700, "y" * 700] * 6)
    ids = [c.id for c in dq.chunk(_doc(text))]
    assert len(ids) == len(set(ids))


# --- retrieval --------------------------------------------------------------------------------


def _mmr_corpus() -> list[dq.Chunk]:
    dup = "Data center revenue rose on Blackwell demand from hyperscale customers."
    lines = [dup, dup + " Again.", dup + " Once more.",
             "Data center revenue from SB Energy leases was capped at $105 billion.",
             "Unrelated text about leases and office buildings in Santa Clara."]
    return [c for n, line in enumerate(lines) for c in dq.chunk(_doc(line, doc_id=f"d{n}"))]


def test_mmr_trades_duplicates_for_a_distinct_relevant_passage() -> None:
    idx = dq.Index(_mmr_corpus())
    top = [c.text for c, _ in idx.search("data center revenue blackwell demand", k=2,
                                         mmr_lambda=1.0, fetch_k=4)]
    mmr = [c.text for c, _ in idx.search("data center revenue blackwell demand", k=2,
                                         mmr_lambda=0.5, fetch_k=4)]
    assert all("Blackwell" in t for t in top)
    assert any("SB Energy" in t for t in mmr)
    assert dq.redundancy(idx, [c for c, _ in idx.search("data center revenue blackwell", k=3,
                                                         mmr_lambda=1.0)]) > 0


def test_search_rejects_fetch_k_below_k_and_returns_nothing_for_no_overlap() -> None:
    idx = dq.Index(_mmr_corpus())
    with pytest.raises(ValueError):
        idx.search("revenue", k=4, fetch_k=2)
    assert idx.search("zzzz qqqq") == []


# --- figures ----------------------------------------------------------------------------------


def test_figures_skip_years_identifiers_and_small_integers() -> None:
    got = [f.text for f in dq.figures(
        "In Q2 of fiscal 2027 the 10-K said H100 revenue was $46.7 billion, up 56%, for 3 "
        "segments, on August 13, 2026, and 42,000 employees.")]
    assert got == ["$46.7 billion", "56%", "42,000"]


@pytest.mark.parametrize(("claim", "passage", "ok"), [
    ("$46.7 billion", "Revenue | $ | 46,743 (in millions)", True),
    ("$46.8 billion", "Revenue | $ | 46,743", False),
    ("56%", "gross margin 55.8 %", True),
    ("56%", "gross margin 55.4 %", False),
    ("56%", "gross margin 56 %", True),
    ("56%", "$56 million of charges", True),
    ("$1.22 billion", "Total revenue | 1,220,068 (in thousands)", True),
    ("$359.5 million", "( 359,468 )", True),
    ("42,000", "approximately 42,000 employees", True),
])
def test_figure_supported_is_scale_aware_and_precision_bound(claim: str, passage: str,
                                                             ok: bool) -> None:
    (fig,) = dq.figures(claim)
    assert dq.figure_supported(fig, passage) is ok


def test_percent_is_not_matched_at_a_scaled_reading() -> None:
    (fig,) = dq.figures("56%")
    assert not dq.figure_supported(fig, "revenue of 0.056 billion")


# --- enforcement ------------------------------------------------------------------------------


def _chunks() -> list[dq.Chunk]:
    return dq.chunk(_doc(FILING, doc_id="NVDA-10-Q-a")) + dq.chunk(
        _doc("Unrelated passage about leases.", doc_id="NVDA-10-K-b", form="10-K"))


def test_enforce_keeps_cited_drops_uncited_and_fabricated_and_strips_bad_keys() -> None:
    a, b = _chunks()
    raw = (f"Data Center revenue was $89.0 billion, up 117% ({a.id}). "
           "NVIDIA is the largest chip company in the world. "
           "Its CEO owns a leather jacket (dqa-deadbeef). "
           f"Gross margin was 75.0% ({a.id}, dqa-0badf00d).")
    out = dq.enforce("q", raw, [a, b])
    assert [c.text for c in out.claims] == ["Data Center revenue was $89.0 billion, up 117%.",
                                            "Gross margin was 75.0%."]
    assert {d.reason for d in out.dropped} == {"uncited", "unresolved_citation"}
    assert out.hallucinated_ids == ["dqa-deadbeef", "dqa-0badf00d"]
    assert out.cited_ids == [a.id]
    assert [c.id for c in out.merely_retrieved] == [b.id]


def test_figure_gate_drops_a_number_the_cited_passage_does_not_carry() -> None:
    a, b = _chunks()
    raw = f"Data Center revenue was $91.0 billion ({a.id}). Revenue was $46.7 billion ({a.id})."
    gated = dq.enforce("q", raw, [a, b])
    assert [c.text for c in gated.claims] == ["Revenue was $46.7 billion."]
    assert gated.dropped[0].reason == "figure_not_in_citation"
    ungated = dq.enforce("q", raw, [a, b], require_figures=False)
    assert len(ungated.claims) == 2


def test_figure_must_be_in_the_passage_cited_not_merely_one_retrieved() -> None:
    a, b = _chunks()
    out = dq.enforce("q", f"Gross margin was 75.0% ({b.id}).", [a, b])
    assert out.refused and out.dropped[0].reason == "figure_not_in_citation"


def test_cannot_answer_is_a_refusal_and_empty_is_too() -> None:
    a, b = _chunks()
    assert dq.enforce("q", "I cannot answer.", [a, b]).refused
    assert dq.enforce("q", "", [a, b]).refused


def test_all_sentences_dropped_is_a_refusal_with_a_reason() -> None:
    a, b = _chunks()
    out = dq.enforce("q", "Revenue was huge. Everyone knows it.", [a, b])
    assert out.refused and "cited" in out.reason and len(out.dropped) == 2


def test_truncated_trailing_sentence_is_dropped() -> None:
    a, b = _chunks()
    out = dq.enforce("q", f"Gross margin was 75.0% ({a.id}). Data Center revenue was ({a.id}",
                     [a, b], truncated=True)
    assert [c.text for c in out.claims] == ["Gross margin was 75.0%."]
    assert out.dropped[-1].reason == "truncated"


def test_sentence_split_respects_abbreviations_and_folds_citation_fragments() -> None:
    a, _ = _chunks()
    parts = dq.split_sentences(
        f"Sales in the U.S. Government segment grew. ({a.id})\n- Inc. filings rose ({a.id}).")
    assert parts[0].startswith("Sales in the U.S. Government segment grew.")
    assert a.id in parts[0]
    assert parts[1].startswith("Inc. filings rose")


def test_key_after_the_full_stop_belongs_to_the_sentence_before_it() -> None:
    """The shape the live model produced on 2026-09-25."""
    a, b = _chunks()
    raw = (f"Data Center revenue was $89.0 billion. ({a.id}) This rose 117% from a year ago. "
           f"({a.id}) It is unrelated. ({b.id})")
    out = dq.enforce("q", raw, [a, b], require_figures=False)
    assert [(c.text, c.cited) for c in out.claims] == [
        ("Data Center revenue was $89.0 billion.", (a.id,)),
        ("This rose 117% from a year ago.", (a.id,)),
        ("It is unrelated.", (b.id,))]


def test_tolerant_citation_parsing_counts_semicolon_and_bracket_forms() -> None:
    a, _ = _chunks()
    out = dq.enforce("q", f"Gross margin was 75.0% [{a.id}; {a.id}].", [a])
    assert out.claims and out.claims[0].cited == (a.id,)


def test_lines_and_sources_number_cited_passages_and_keep_retrieved_apart() -> None:
    a, b = _chunks()
    out = dq.enforce("q", f"Gross margin was 75.0% ({a.id}).", [a, b])
    assert out.lines == ["Gross margin was 75.0%.[1]"]
    assert all(isinstance(s, Source) and s.kind == "evidence" for s in out.sources)
    assert out.sources[0].ref.endswith(f"#{a.id}") and out.sources[0].detail.startswith("[1]")
    assert [s.ref for s in out.retrieved_sources] == [f"{b.doc.url}#{b.id}"]
    blob = out.as_dict()
    assert blob["cited"] == [a.id] and blob["merely_retrieved"] == [b.id]


# --- end to end with a scripted model ---------------------------------------------------------


def _answer_from_prompt(messages: list[dict[str, Any]]) -> str:
    ids = re.findall(r"\[(dqa-[0-9a-f]{8})\]", messages[-1]["content"])
    return (f"Data Center revenue was $89.0 billion, up 117% ({ids[0]}). "
            "Guessing more (dqa-ffff0000).")


def test_answer_retrieves_prompts_and_enforces() -> None:
    model = Scripted(_answer_from_prompt)
    got = dq.answer("What was Data Center revenue?", _chunks(), model)
    assert len(model.prompts) == 1
    prompt = model.prompts[0][-1]["content"]
    assert dq.CANNOT_ANSWER in prompt and "dqa-" in prompt
    assert [c.text for c in got.claims] == ["Data Center revenue was $89.0 billion, up 117%."]
    assert got.hallucinated_ids == ["dqa-ffff0000"]


def test_answer_without_a_matching_passage_refuses_without_calling_the_model() -> None:
    model = Scripted("unused")
    got = dq.answer("zzzz qqqq", _chunks(), model)
    assert got.refused and model.prompts == []


def test_research_answer_returns_lines_sources_and_data_for_the_console() -> None:
    model = Scripted(_answer_from_prompt)
    lines, sources, data = dq.research_answer("What was Data Center revenue?", "NVDA", model,
                                              documents=[_doc(FILING)])
    assert lines[0].startswith("From NVDA 10-Q filed 26 Aug 2026")
    assert lines[1].endswith("[1]") and len(sources) == 1
    assert data["hallucinated_ids"] == ["dqa-ffff0000"]


def test_research_answer_with_no_documents_says_so() -> None:
    class Empty:
        def latest(self, ticker: str) -> list[dq.Document]:
            return []

    lines, sources, data = dq.research_answer("q", "zzz", Scripted("x"),
                                              source=Empty())  # type: ignore[arg-type]
    assert "No 10-K" in lines[0] and sources == [] and data["refused"]


def test_headlines_become_one_chunk_documents() -> None:
    from datetime import datetime

    from argus.market.evidence import Headline

    h = Headline(feed="cnbc", title="Nvidia to buy Hugging Face for $11.9 billion",
                 link="https://example.com/a", published=datetime(2026, 9, 3, 12, 0))
    (doc,) = dq.headline_documents("nvda", [h])
    assert doc.form == "headline" and doc.ticker == "NVDA" and len(dq.chunk(doc)) == 1


# --- the eval ---------------------------------------------------------------------------------


def test_gold_matching_is_scale_aware_for_figures_and_literal_for_names() -> None:
    assert ev.gold_met("$1.22 billion", "total revenue of $1,220 million")
    assert ev.gold_met("16 percent", "up 16% year over year")
    assert not ev.gold_met("$89.0 billion", "revenue of $88.0 billion")
    assert ev.gold_met("Anthony Armstrong", "appointed anthony  armstrong")
    assert not ev.gold_met("first half of 2027", "second half of 2027")


def test_deterministic_verdict() -> None:
    passage = "Gross margin increased to 75.0% due to improved mix from Blackwell Ultra."
    assert ev.deterministic_verdict("Gross margin increased to 75.0%.", passage) == "supported"
    assert ev.deterministic_verdict("Gross margin increased to 76.0%.", passage) == "unsupported"
    assert ev.deterministic_verdict("Operating expenses soared across Europe.",
                                    passage) == "unknown"


def test_capped_model_serves_cache_and_refuses_past_the_cap(tmp_path: Path) -> None:
    inner = Scripted("Answer.")
    path = tmp_path / "cache.json"
    m = ev.CappedModel(inner, cap=1, cache_path=path)
    msg = [{"role": "user", "content": "one"}]
    assert m.complete(msg).content == "Answer."
    assert m.complete(msg).content == "Answer." and m.live == 1 and m.cached == 1
    with pytest.raises(ev.CallCapReached):
        m.complete([{"role": "user", "content": "two"}])
    again = ev.CappedModel(None, cap=0, cache_path=path)
    assert again.complete(msg).content == "Answer." and again.live == 0


def test_policies_differ_exactly_where_they_should() -> None:
    a, b = _chunks()
    raw = (f"Gross margin was 75.0% ({a.id}). Revenue was $91.0 billion ({a.id}). "
           "It is great (dqa-deadbeef). It is big.")
    got = ev.sentences_of("q", raw, [a, b], truncated=False)
    kept = {p: [s.text for s in got if s.kept_by[p]] for p in ev.POLICIES}
    assert len(kept["raw"]) == len(kept["paper-qa rule"]) == 4
    assert kept["citation enforced"] == ["Gross margin was 75.0%.", "Revenue was $91.0 billion."]
    assert kept["citation + figure gate"] == ["Gross margin was 75.0%."]
    assert got[2].fabricated == ("dqa-deadbeef",)


def test_questions_are_well_formed_and_their_evidence_is_in_the_frozen_corpus() -> None:
    docs = ev.load_corpus()
    assert len(ev.QUESTIONS) >= 20 and len({q.ticker for q in ev.QUESTIONS}) >= 3
    assert len({q.qid for q in ev.QUESTIONS}) == len(ev.QUESTIONS)
    text = {t: ev._norm("\n".join(d.text for d in docs if d.ticker == t)) for t in ev.TICKERS}
    for q in ev.QUESTIONS:
        if q.answerable:
            assert q.gold and q.evidence, q.qid
            assert all(ev._norm(e) in text[q.ticker] for e in q.evidence), q.qid
        else:
            assert not q.gold


def test_full_eval_runs_on_the_frozen_corpus_with_a_scripted_model(tmp_path: Path) -> None:
    def reply(messages: list[dict[str, Any]]) -> str:
        content = messages[-1]["content"]
        if "Return JSON" in content:
            ids = re.findall(r"### (\S+)", content)
            return '{"verdicts": [' + ",".join(
                f'{{"id": "{i}", "verdict": "SUPPORTED"}}' for i in ids) + "]}"
        first = re.search(r"\[(dqa-[0-9a-f]{8})\][^\n]*\n([^\n]+)", content)
        assert first is not None
        line = first.group(2)[:120].rstrip(".")
        return f"{line} ({first.group(1)}). Made-up claim (dqa-00000000)."

    model = ev.CappedModel(Scripted(reply), cap=ev.CALL_CAP, cache_path=tmp_path / "c.json")
    report = ev.run(model)
    assert report["questions"] == len(ev.QUESTIONS)
    assert report["qwen"]["live_calls_this_run"] <= ev.CALL_CAP
    assert report["fabricated_citation_ids"] == len(ev.QUESTIONS)
    enforced = report["audit_model"]["citation enforced"]
    assert enforced["uncited"] == 0
    assert report["audit_model"]["paper-qa rule"]["uncited"] >= 1
    assert set(report["retrieval"]) == {f"k={k},lambda={lam}" for k in ev.KS for lam in ev.LAMBDAS}
