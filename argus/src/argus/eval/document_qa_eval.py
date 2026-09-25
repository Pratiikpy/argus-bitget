"""Does `research/document_qa.py` keep only claims its citations carry? Measured on real filings.

**The corpus.** The latest 10-K, 10-Q and 8-K (with exhibit 99.1 where one exists) of NVDA, AAPL,
MSFT, TSLA and COIN, read from SEC EDGAR on 2026-09-25 and frozen under
``data/document_qa_corpus/`` with a SHA-256 per document, so every number below can be reproduced
on the same text. :data:`QUESTIONS` is thirty questions written after reading those filings: 27
whose answer is stated in them, each with the figures or names a correct answer must contain and
the passage that states it, and 3 that the filings do not answer (a quarter not yet reported, an
analyst price target, a guidance figure the company does not give).

**What is measured, and how each measurement avoids grading itself.**

* *Retrieval, separately from answering* — the Mind2Web split (`metric.py:99-113`): does the
  retrieved set contain the passage that states the answer, at k of 8, 10 and 12 and MMR lambda
  1.0 (plain top-k), 0.85, 0.7 and 0.5, and how many retrieved pairs are near-duplicates.
  Deterministic, no model call. ``document_qa.TOP_K`` and ``MMR_LAMBDA`` are set from this table,
  on the same questions — so the choice is in-sample and says so where it is made.
* *Answering* — one Qwen call per question, answers cached to ``data/document_qa_responses.json``
  so a re-run spends nothing. The same raw answer is then put through four policies:

  - ``raw``: every sentence as the model wrote it — what a reader gets with no enforcement;
  - ``paper-qa rule``: fabricated keys stripped, the sentence kept (`paperqa/types.py:510-515`);
  - ``citation enforced``: a sentence with no resolving citation is dropped;
  - ``citation + figure gate``: also drops a sentence whose figure is in none of its cited
    passages — the module's default.
* *A FACT-style citation audit of every claim each policy keeps.* Two checkers:

  - **deterministic**: every figure in the claim is found in the passages it cites (at any
    reporting scale, to the claim's precision) and at least 60% of its content words appear
    there. This checker shares its figure test with the figure gate, so for the gated policy its
    figure half is true by construction — it is reported, and it is not the evidence;
  - **model check** (the evidence): Qwen reads the claim and only the passages it cites and says
    SUPPORTED or NOT_SUPPORTED. It never sees the gate's decision. An uncited claim has no passage
    to check, and is counted as ``uncited``, not as supported.
* *Correctness*: a kept answer is correct when it states every gold figure (same scale-aware
  matching) and every gold name; the three unanswerable questions are correct when refused.

* *Starved retrieval* (:func:`starved`): twelve answerable questions re-asked with every chunk
  that holds the answer's evidence string withheld — the condition where a model is tempted to
  answer from memory under a real citation.

**Qwen cap: 60 calls**, enforced by :class:`CappedModel` across runs (every live call leaves one
cache entry, and the entries already made come off the cap). Spent: 54 — 30 answers and 9 judge
batches on the main set, 12 answers and 3 judge batches starved.

**Results, 2026-09-25 (``data/document_qa_eval.json``) — honest reading: the enforcement tied.**

* Retrieval: the answer passage was in the retrieved set for 26 of 27 questions at the default
  (k=12, lambda 0.85), against 25 for plain top-k at k=12 and 23 at k=8.
* Answers: 25 of 27 correct, 3 of 3 unanswerable refused. The two misses are honest ones — one
  answer left out the growth rate, and one question (Tesla's bitcoin) whose passage was not
  retrieved was refused rather than guessed.
* **Qwen 3.8 fabricated no citation key in 42 answers**, and put a key on every sentence. The
  model check found all 71 main-set claims supported under every policy, so "citation enforced"
  and the "paper-qa rule" kept exactly what "raw" kept: on this model and these questions the
  enforcement caught nothing, because there was nothing to catch. Its value here is a guarantee
  (proved on adversarial scripted outputs in `tests/test_document_qa.py`), not a measured gain.
* The figure gate dropped one main-set sentence, a difference the model computed itself
  ("a decrease of $277,140 thousand", 1,497,208 - 1,220,068) that no passage states. The model
  check called it supported; the gate enforces the prompt's own rule against computed figures.
  Counted as a false drop against the gate, since the arithmetic is right.
* Starved: 4 of 12 refused; the other 8 answered from *other* passages that state the same fact
  (a table row, a later note), 26 of 27 claims supported. The one unsupported claim (Apple's
  Greater China iPhone mix "compared to other segments") is an interpretive overreach with no
  figure in it — it passed every policy. **No policy here catches a wrong reading of a real
  passage**; only the figure gate catches a wrong number.

**Not done here, stated rather than implied.** The synthesis plan (`research/mypr-teardowns/
_SYNTHESIS.md`, row M2) names a head-to-head with gpt-researcher and paper-qa themselves on the
same questions. Neither was run: both need an embedding API and their own LLM wiring (litellm /
LangChain), which would put their calls outside the 60-call cap. The ``paper-qa rule`` policy
above is paper-qa's *citation-stripping behaviour* reproduced on our answers, not paper-qa run.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from argus.eval.artefact import write
from argus.llm.qwen import Completion, QwenError, Thinking, Usage, extract_json_object
from argus.research import document_qa as dq

DATA = Path(__file__).resolve().parents[3] / "data"
CORPUS = DATA / "document_qa_corpus"
MANIFEST = CORPUS / "manifest.json"
RESPONSES = DATA / "document_qa_responses.json"
REPORT = DATA / "document_qa_eval.json"
TICKERS = ("NVDA", "AAPL", "MSFT", "TSLA", "COIN")
CALL_CAP = 60
LAMBDAS = (1.0, 0.85, 0.7, 0.5)
KS = (8, 10, 12)
JUDGE_BATCH = 3
OVERLAP_SUPPORTED = 0.6


@dataclass(frozen=True)
class Question:
    qid: str
    ticker: str
    text: str
    gold: tuple[str, ...] = ()
    """Figures (``"$89.0 billion"``, ``"117%"``) or names a correct answer must state."""
    evidence: tuple[str, ...] = ()
    """Verbatim substrings of the filing that state the answer; retrieval recall needs them all."""
    answerable: bool = True


QUESTIONS: tuple[Question, ...] = (
    Question("nvda-dc", "NVDA", "What was NVIDIA's Data Center revenue in the second quarter of "
             "fiscal 2027, and how did it change from a year ago?",
             ("$89.0 billion", "117%"), ("Data Center revenue was $89.0 billion, up 117%",)),
    Question("nvda-gm", "NVDA", "What was NVIDIA's gross margin in the second quarter of fiscal "
             "2027 and what drove the change from a year earlier?",
             ("75.0%", "Blackwell Ultra"), ("Gross margin increased to 75.0%",)),
    Question("nvda-buyback", "NVDA", "How much common stock did NVIDIA repurchase in the second "
             "quarter of fiscal 2027, and how much repurchase authorization remained?",
             ("$19.7 billion", "$99.3 billion"),
             ("$ 19.7 billion", "repurchase up to $ 99.3 billion")),
    Question("nvda-sbenergy", "NVDA", "What guarantees did NVIDIA enter into with SB Energy "
             "in August 2026?", ("$105 billion",), ("capped at a total of $ 105 billion",)),
    Question("nvda-hf", "NVDA", "What is NVIDIA paying to acquire Hugging Face and when is the "
             "deal expected to close?", ("$11.9 billion", "first half of 2027"),
             ("approximately $11.9 billion purchase price",)),
    Question("nvda-staff", "NVDA", "How many employees did NVIDIA have at the end of fiscal year "
             "2026?", ("42,000",), ("we had approximately 42,000 employees",)),
    Question("nvda-supply", "NVDA", "How large were NVIDIA's supply commitments as of July 26, "
             "2026, and what were they the previous quarter?", ("$279 billion", "$119 billion"),
             ("from $ 119 billion last quarter to $ 279 billion",)),
    Question("aapl-rev", "AAPL", "What revenue did Apple report for its fiscal 2026 third quarter "
             "and how much did it grow year over year?", ("$109.4 billion", "16 percent"),
             ("quarterly revenue of $109.4 billion, up 16 percent",)),
    Question("aapl-gm", "AAPL", "What was Apple's gross margin in the June 2026 quarter and how "
             "did tariff refunds affect it?", ("50.1 percent", "2 percentage points"),
             ("gross margin was 50.1 percent",)),
    Question("aapl-div", "AAPL", "What dividend did Apple's board declare with its third quarter "
             "fiscal 2026 results, and when is it payable?", ("$0.27", "August 13, 2026"),
             ("cash dividend of $0.27 per share",)),
    Question("aapl-china", "AAPL", "How did Apple's Greater China net sales change in the third "
             "quarter of fiscal 2026?", ("22%",), ("Greater China | 18,816 | 15,369 | 22%",)),
    Question("aapl-buyback", "AAPL", "How much common stock did Apple repurchase during the third "
             "quarter of fiscal 2026?", ("$25.8 billion",),
             ("the Company repurchased $25.8 billion of its common stock",)),
    Question("aapl-staff", "AAPL", "How many full-time equivalent employees did Apple have at the "
             "end of fiscal 2025?", ("166,000",),
             ("approximately 166,000 full-time equivalent employees",)),
    Question("msft-segments", "MSFT", "What reporting segments will Microsoft use beginning in "
             "fiscal 2027?", ("Agents and Infra", "Devices and Consumer"),
             ("two segments: Agents and Infra and Devices and Consumer",)),
    Question("msft-cloud-fy", "MSFT", "How much Microsoft Cloud revenue did Microsoft report for "
             "fiscal year 2026 and how fast did it grow?", ("$214.4 billion", "27%"),
             ("Microsoft Cloud revenue increased 27% to $214.4 billion",)),
    Question("msft-staff", "MSFT", "How many people did Microsoft employ full time as of June 30, "
             "2026?", ("223,000",), ("employed approximately 223,000 people",)),
    Question("msft-cloud-q", "MSFT", "What was Microsoft Cloud revenue in the quarter ended "
             "March 31, 2026, and its growth rate?", ("$54.5 billion", "29%"),
             ("Microsoft Cloud revenue increased 29% to $54.5 billion",)),
    Question("tsla-rev", "TSLA", "What were Tesla's total revenues in the second quarter of 2026 "
             "and how much did they grow year over year?", ("$28.24 billion", "26%"),
             ("total revenues of $28.24 billion",)),
    Question("tsla-deliv", "TSLA", "How many Model 3 and Model Y vehicles did Tesla deliver in the "
             "second quarter of 2026?", ("467,762",), ("Model 3/Y deliveries",)),
    Question("tsla-btc", "TSLA", "How much Bitcoin does Tesla hold according to its latest "
             "quarterly report?", ("11,509", "$386 million"), ("11,509 units of Bitcoin",)),
    Question("tsla-energy", "TSLA", "How did Tesla's energy generation and storage revenue change "
             "in the second quarter of 2026 and why?", ("$350 million", "13%", "Megapack"),
             ("Energy generation and storage revenue increased $350 million, or 13%",)),
    Question("tsla-staff", "TSLA", "What was Tesla's worldwide employee headcount at the end of "
             "2025?", ("134,785",), ("employee headcount worldwide was 134,785",)),
    Question("coin-rev", "COIN", "What was Coinbase's total revenue in the second quarter of 2026 "
             "compared with the second quarter of 2025?", ("$1.22 billion", "$1.50 billion"),
             ("Total revenue | $ | 1,220,068 | $ | 1,497,208",)),
    Question("coin-board", "COIN", "Who did Coinbase appoint to its board of directors in "
             "September 2026, and to which committee?",
             ("Anthony Armstrong", "Audit and Compliance Committee"),
             ("appointed Anthony Armstrong to serve as a director",)),
    Question("coin-netloss", "COIN", "What net income or loss did Coinbase report for the three "
             "months ended June 30, 2026?", ("$359.5 million",),
             ("Net (loss) income | $ | ( 359,468 )",)),
    Question("coin-concentration", "COIN", "How concentrated was Coinbase's revenue in a single "
             "counterparty in the second quarter of 2026?", ("26%",),
             ("one counterparty accounted for 26 %",)),
    Question("coin-staff", "COIN", "How many employees did Coinbase have at the end of 2025?",
             ("4,951",), ("we had 4,951 employees",)),
    Question("tsla-q3", "TSLA", "What total revenue did Tesla report for the third quarter of "
             "2026?", answerable=False),
    Question("coin-target", "COIN", "What price target do analysts have on Coinbase stock?",
             answerable=False),
    Question("aapl-guide", "AAPL", "What revenue does Apple guide for its fiscal 2027 first "
             "quarter, in dollars?", answerable=False),
)


# --- corpus -----------------------------------------------------------------------------------


def _doc_to_row(d: dq.Document) -> dict[str, Any]:
    return {"doc_id": d.doc_id, "ticker": d.ticker, "form": d.form, "filed": d.filed.isoformat(),
            "url": d.url, "title": d.title, "sha256": d.sha256, "chars": len(d.text)}


def freeze_corpus(source: dq.EdgarDocuments | None = None) -> list[dq.Document]:
    """Read every ticker's latest filings from EDGAR and freeze them to disk with hashes."""
    src = source or dq.EdgarDocuments()
    docs = [d for t in TICKERS for d in src.latest(t)]
    CORPUS.mkdir(parents=True, exist_ok=True)
    for d in docs:
        (CORPUS / f"{d.doc_id}.txt.gz").write_bytes(
            gzip.compress(d.text.encode("utf-8"), mtime=0))
    write(MANIFEST, {"fetched": date.today().isoformat(), "source": "SEC EDGAR",
                     "documents": [_doc_to_row(d) for d in docs]})
    return docs


def load_corpus() -> list[dq.Document]:
    """The frozen corpus, each document checked against its recorded SHA-256."""
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]
    docs: list[dq.Document] = []
    for r in rows:
        text = gzip.decompress((CORPUS / f"{r['doc_id']}.txt.gz").read_bytes()).decode("utf-8")
        doc = dq.Document(doc_id=r["doc_id"], ticker=r["ticker"], form=r["form"],
                          filed=date.fromisoformat(r["filed"]), url=r["url"], title=r["title"],
                          text=text)
        if doc.sha256 != r["sha256"]:
            raise ValueError(f"{r['doc_id']}: corpus text does not match its recorded hash")
        docs.append(doc)
    return docs


# --- the capped, cached model -----------------------------------------------------------------


class CallCapReached(QwenError):
    """The eval's hard cap on live model calls would be exceeded."""


class CappedModel:
    """A `ChatModel` that answers from a disk cache first and refuses a live call past the cap.

    ``live`` counts calls that reached the endpoint in this run; ``cached`` counts answers served
    from ``data/document_qa_responses.json``. The cache key is a hash of the exact messages, so an
    edited prompt is a new call, never a stale answer.
    """

    def __init__(self, inner: Any | None, *, cap: int = CALL_CAP,
                 cache_path: Path = RESPONSES) -> None:
        self._inner = inner
        self.budget = getattr(inner, "budget", None)
        self.cap = cap
        self.live = 0
        self.cached = 0
        self._path = cache_path
        self._cache: dict[str, dict[str, Any]] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {})

    @property
    def entries(self) -> int:
        """Every live call this eval has ever made: each one left exactly one cache entry."""
        return len(self._cache)

    @staticmethod
    def key(messages: list[dict[str, Any]]) -> str:
        return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        k = self.key(messages)
        hit = self._cache.get(k)
        if hit is not None:
            self.cached += 1
            return Completion(content=hit["content"], reasoning="", finish_reason=hit["finish"],
                              usage=Usage(0, 0, 0, 0, reported=False))
        if self._inner is None:
            raise CallCapReached("no live model configured and the answer is not cached")
        if self.live >= self.cap:
            raise CallCapReached(f"live call cap of {self.cap} reached")
        self.live += 1
        got: Completion = self._inner.complete(messages, **kwargs)
        self._cache[k] = {"content": got.content, "finish": got.finish_reason,
                          "prompt_tokens": got.usage.prompt_tokens,
                          "completion_tokens": got.usage.completion_tokens,
                          "reasoning_tokens": got.usage.reasoning_tokens}
        write(self._path, self._cache)
        return got

    def complete_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        got = self.complete(messages, json_mode=True, max_tokens=kwargs.get("max_tokens", 4000),
                            thinking=Thinking.LOW)
        parsed: dict[str, Any] = json.loads(extract_json_object(got.content))
        return parsed


# --- retrieval --------------------------------------------------------------------------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def recall(retrieved: Sequence[dq.Chunk], evidence: Sequence[str]) -> bool:
    blob = _norm("\n".join(c.text for c in retrieved))
    return all(_norm(e) in blob for e in evidence)


def retrieval_table(indexes: dict[str, dq.Index]) -> dict[str, Any]:
    """Recall of the answer passage and near-duplicate share, for every (k, lambda) in the grid."""
    rows: dict[str, Any] = {}
    answerable = [q for q in QUESTIONS if q.answerable]
    for k in KS:
        for lam in LAMBDAS:
            hits, dup = 0, 0.0
            missed: list[str] = []
            for q in answerable:
                got = [c for c, _ in indexes[q.ticker].search(q.text, k=k, mmr_lambda=lam)]
                if recall(got, q.evidence):
                    hits += 1
                else:
                    missed.append(q.qid)
                dup += dq.redundancy(indexes[q.ticker], got)
            rows[f"k={k},lambda={lam}"] = {
                "recall": round(hits / len(answerable), 4), "hits": hits,
                "questions": len(answerable),
                "near_duplicate_pair_share": round(dup / len(answerable), 4), "missed": missed}
    return rows


# --- auditing ---------------------------------------------------------------------------------


def gold_met(gold: str, text: str) -> bool:
    """A pure figure (``"$89.0 billion"``, ``"16 percent"``) is matched scale-aware, so
    ``$89,000 million`` satisfies it; anything with words (a name, a date) must appear verbatim,
    case- and whitespace-insensitively."""
    residue = re.sub(r"\b(?:percent|billion|million|thousand)\b|[\s$\d.,%]", "", gold)
    figs = dq.figures(gold)
    if figs and not residue:
        return all(dq.figure_supported(f, text) for f in figs)
    return _norm(gold) in _norm(text)


def deterministic_verdict(claim: str, cited_text: str) -> str:
    """``supported`` | ``unsupported`` | ``unknown`` by figure and content-word overlap."""
    if any(not dq.figure_supported(f, cited_text) for f in dq.figures(claim)):
        return "unsupported"
    words = {t for t in dq.tokens(claim) if not re.fullmatch(r"[\d.]+", t)}
    if not words:
        return "supported"
    have = set(dq.tokens(cited_text))
    share = len(words & have) / len(words)
    return "supported" if share >= OVERLAP_SUPPORTED else "unknown"


@dataclass
class Sentence:
    """One raw sentence of a model answer, with what every policy does to it."""

    sid: str
    text: str
    resolved: tuple[str, ...]
    fabricated: tuple[str, ...]
    kept_by: dict[str, bool]


POLICIES = ("raw", "paper-qa rule", "citation enforced", "citation + figure gate")


def sentences_of(qid: str, raw: str, retrieved: Sequence[dq.Chunk],
                 truncated: bool) -> list[Sentence]:
    """Split the raw answer and record, per sentence, which policy keeps it."""
    ungated = dq.enforce(qid, raw, retrieved, require_figures=False, truncated=truncated)
    gated = dq.enforce(qid, raw, retrieved, require_figures=True, truncated=truncated)
    kept_ungated = {c.text for c in ungated.claims}
    kept_gated = {c.text for c in gated.claims}
    by_id = {c.id for c in retrieved}
    out: list[Sentence] = []
    if ungated.refused and not ungated.claims and dq.CANNOT_ANSWER.lower() in raw.lower():
        return out
    for n, s in enumerate(dq.split_sentences(raw)):
        body = dq.strip_citations(s)
        if not re.search(r"[A-Za-z0-9]", body):
            continue
        ids = dq.citation_ids(s)
        out.append(Sentence(
            sid=f"{qid}-s{n + 1}", text=body,
            resolved=tuple(i for i in ids if i in by_id),
            fabricated=tuple(i for i in ids if i not in by_id),
            kept_by={"raw": True, "paper-qa rule": True,
                     "citation enforced": body in kept_ungated,
                     "citation + figure gate": body in kept_gated}))
    return out


JUDGE_SYSTEM = (
    "You audit claims against source passages. For each claim, decide only from the passages "
    "listed with it whether they state what the claim says. A claim is SUPPORTED only if every "
    "figure, name and relationship in it is stated in those passages; otherwise NOT_SUPPORTED. "
    "Do not use outside knowledge."
)


def judge_prompt(items: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """``items``: (sentence id, claim, cited passages text)."""
    body = "\n\n".join(f"### {sid}\nClaim: {claim}\nPassages:\n{passages}"
                       for sid, claim, passages in items)
    user = (f"{body}\n\nReturn JSON: {{\"verdicts\": [{{\"id\": \"<sentence id>\", "
            f"\"verdict\": \"SUPPORTED\" or \"NOT_SUPPORTED\"}}, ...]}} with one entry per claim.")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def rates(sentences: Sequence[Sentence], policy: str, verdict: Callable[[Sentence], str]
          ) -> dict[str, Any]:
    kept = [s for s in sentences if s.kept_by[policy]]
    counts = {"supported": 0, "unsupported": 0, "unknown": 0, "uncited": 0}
    for s in kept:
        counts["uncited" if not s.resolved else verdict(s)] += 1
    n = len(kept)
    return {"kept": n, **counts,
            **{f"{k}_rate": (round(v / n, 4) if n else None) for k, v in counts.items()}}


def judge(model: CappedModel, groups: Sequence[Sequence[Sentence]], chunk_text: dict[str, str]
          ) -> tuple[dict[str, str], int, list[str]]:
    """Model verdicts for every sentence that cites a real passage, JUDGE_BATCH groups a call.

    A batch that fails (cap reached, unparseable JSON) is recorded, and its sentences stay
    ``unknown`` rather than being guessed."""
    verdicts: dict[str, str] = {}
    cited = [g for g in groups if any(s.resolved for s in g)]
    attempted = 0
    errors: list[str] = []
    for start in range(0, len(cited), JUDGE_BATCH):
        items = [(s.sid, s.text, "\n---\n".join(chunk_text[i] for i in s.resolved))
                 for g in cited[start:start + JUDGE_BATCH] for s in g if s.resolved]
        attempted += 1
        try:
            got = model.complete_json(judge_prompt(items))
        except (QwenError, ValueError) as exc:
            errors.append(f"batch {start // JUDGE_BATCH}: {exc}")
            continue
        for v in got.get("verdicts", []):
            if isinstance(v, dict) and isinstance(v.get("id"), str):
                verdicts[v["id"]] = ("supported" if str(v.get("verdict", "")).upper()
                                     == "SUPPORTED" else "unsupported")
    return verdicts, attempted, errors


STARVED = 12
"""Questions in the starved condition: the first twelve answerable ones in :data:`QUESTIONS`
order, fixed before any starved answer was seen."""


def starved(model: CappedModel, indexes: dict[str, dq.Index], chunk_text: dict[str, str]
            ) -> dict[str, Any]:
    """The adversarial condition: the passage that states the answer is withheld.

    On the plain set the model can simply read the answer, so the enforcement has little to do.
    Here every chunk containing a question's evidence string is removed before retrieval, so the
    prompt holds related passages but not the answer. The right outcome is a refusal; the failure
    being tested for is an answer from memory or from a neighbouring figure, dressed in a real
    citation. Each policy's kept claims are audited by the same model check.
    """
    picked = [q for q in QUESTIONS if q.answerable][:STARVED]
    groups: list[list[Sentence]] = []
    rows: dict[str, Any] = {}
    errors: list[str] = []
    for q in picked:
        idx = indexes[q.ticker]
        pool = [c for c in idx.chunks
                if not any(_norm(e) in _norm(c.text) for e in q.evidence)]
        retrieved = [c for c, _ in dq.Index(pool).search(q.text)]
        try:
            completion = model.complete(dq.build_prompt(q.text, retrieved), temperature=0.0,
                                        max_tokens=dq.MAX_TOKENS, thinking=Thinking.LOW)
        except QwenError as exc:
            errors.append(f"{q.qid}: {exc}")
            continue
        truncated = completion.finish_reason == "length"
        sid = f"starved-{q.qid}"
        sentences = sentences_of(sid, completion.content, retrieved, truncated)
        groups.append(sentences)
        gated = dq.enforce(q.text, completion.content, retrieved, truncated=truncated)
        rows[q.qid] = {
            "raw_answer": completion.content,
            "model_refused": not sentences,
            "shown_after_enforcement": [c.text for c in gated.claims],
            "raw_states_gold": bool(sentences) and any(
                gold_met(g, " ".join(s.text for s in sentences)) for g in q.gold),
        }
    verdicts, attempted, judge_errors = judge(model, groups, chunk_text)

    def mod(s: Sentence) -> str:
        return verdicts.get(s.sid, "unknown")

    flat = [s for g in groups for s in g]
    return {
        "questions": len(picked),
        "answered": len(rows),
        "model_refused": sum(r["model_refused"] for r in rows.values()),
        "empty_after_enforcement": sum(not r["shown_after_enforcement"] for r in rows.values()),
        "raw_states_a_gold_figure": sum(r["raw_states_gold"] for r in rows.values()),
        "fabricated_citation_ids": sum(len(s.fabricated) for s in flat),
        "audit_model": {p: rates(flat, p, mod) for p in POLICIES},
        "judge_calls_attempted": attempted,
        "errors": errors + judge_errors,
        "per_question": rows,
    }


def run(model: CappedModel, *, refresh_corpus: bool = False) -> dict[str, Any]:
    docs = freeze_corpus() if refresh_corpus or not MANIFEST.exists() else load_corpus()
    indexes = {t: dq.Index([c for d in docs if d.ticker == t for c in dq.chunk(d)])
               for t in TICKERS}
    retrieval = retrieval_table(indexes)

    answers: dict[str, dict[str, Any]] = {}
    all_sentences: dict[str, list[Sentence]] = {}
    chunk_text: dict[str, str] = {c.id: c.text for idx in indexes.values() for c in idx.chunks}
    for q in QUESTIONS:
        retrieved = [c for c, _ in indexes[q.ticker].search(q.text)]
        messages = dq.build_prompt(q.text, retrieved)
        completion = model.complete(messages, temperature=0.0, max_tokens=dq.MAX_TOKENS,
                                    thinking=Thinking.LOW)
        truncated = completion.finish_reason == "length"
        gated = dq.enforce(q.text, completion.content, retrieved, truncated=truncated)
        ungated = dq.enforce(q.text, completion.content, retrieved, require_figures=False,
                             truncated=truncated)
        sentences = sentences_of(q.qid, completion.content, retrieved, truncated)
        all_sentences[q.qid] = sentences
        kept_text = " ".join(c.text for c in gated.claims)
        raw_text = " ".join(s.text for s in sentences)
        if q.answerable:
            correct_gated = (not gated.refused) and all(gold_met(g, kept_text) for g in q.gold)
            correct_raw = bool(sentences) and all(gold_met(g, raw_text) for g in q.gold)
        else:
            correct_gated = gated.refused
            correct_raw = not sentences
        answers[q.qid] = {
            "ticker": q.ticker, "question": q.text, "answerable": q.answerable,
            "gold": list(q.gold), "retrieval_hit": recall(retrieved, q.evidence)
            if q.answerable else None,
            "raw_answer": completion.content, "truncated": truncated,
            "correct_with_enforcement": correct_gated, "correct_raw": correct_raw,
            "enforced": gated.as_dict(),
            "ungated_dropped": [d.reason for d in ungated.dropped],
        }

    # Deterministic audit.
    def det(s: Sentence) -> str:
        return deterministic_verdict(s.text, "\n".join(chunk_text[i] for i in s.resolved))

    flat = [s for q in QUESTIONS for s in all_sentences[q.qid]]
    deterministic = {p: rates(flat, p, det) for p in POLICIES}

    # Model audit, batched by question, only for sentences that cite a real passage.
    verdicts, judge_calls_attempted, judge_errors = judge(
        model, [all_sentences[q.qid] for q in QUESTIONS], chunk_text)

    def mod(s: Sentence) -> str:
        return verdicts.get(s.sid, "unknown")

    judged = {p: rates(flat, p, mod) for p in POLICIES}
    gate_dropped = [s for s in flat if s.kept_by["citation enforced"]
                    and not s.kept_by["citation + figure gate"]]
    answerable = [a for a in answers.values() if a["answerable"]]
    unanswerable = [a for a in answers.values() if not a["answerable"]]
    fabricated = sum(len(s.fabricated) for s in flat)
    stress = starved(model, indexes, chunk_text)  # before the call counts below are read
    return {
        "generated": date.today().isoformat(),
        "corpus": {"documents": [_doc_to_row(d) for d in docs],
                   "chunks_per_ticker": {t: len(indexes[t].chunks) for t in TICKERS}},
        "questions": len(QUESTIONS),
        "answerable": len(answerable),
        "retrieval": retrieval,
        "mmr_lambda_used": dq.MMR_LAMBDA,
        "top_k_used": dq.TOP_K,
        "answer_correct": {
            "with_enforcement": sum(a["correct_with_enforcement"] for a in answerable),
            "raw": sum(a["correct_raw"] for a in answerable),
            "of": len(answerable),
            "refused_when_unanswerable": sum(a["correct_with_enforcement"] for a in unanswerable),
            "of_unanswerable": len(unanswerable),
        },
        "sentences": len(flat),
        "fabricated_citation_ids": fabricated,
        "audit_deterministic": deterministic,
        "audit_model": judged,
        "figure_gate": {
            "dropped": len(gate_dropped),
            "dropped_that_model_judged_supported": sum(mod(s) == "supported"
                                                       for s in gate_dropped),
            "dropped_sentences": [{"id": s.sid, "text": s.text, "model": mod(s)}
                                  for s in gate_dropped],
        },
        "model_verdicts_obtained": len(verdicts),
        "judge_errors": judge_errors,
        "qwen": {"live_calls_this_run": model.live, "served_from_cache": model.cached,
                 "live_calls_ever": model.entries,
                 "cap": model.cap, "judge_calls_attempted": judge_calls_attempted},
        "starved": stress,
        "answers": answers,
    }


def main() -> int:  # pragma: no cover - CLI, real LLM cost
    inner: Any = None
    if os.environ.get("BITGET_QWEN_API_KEY"):
        from argus.llm.qwen import QwenClient, TokenBudget
        inner = QwenClient(budget=TokenBudget(limit=400_000))
    model = CappedModel(inner)
    # The cap is on the eval, not on one invocation: calls already made (one cache entry each)
    # come off it, so an interrupted and resumed run cannot spend more than CALL_CAP in total.
    model.cap = max(0, CALL_CAP - model.entries)
    report = run(model, refresh_corpus="--refresh" in sys.argv)
    write(REPORT, report)
    print(json.dumps({k: report[k] for k in ("retrieval", "answer_correct",
                                             "fabricated_citation_ids", "audit_model",
                                             "figure_gate", "qwen")}, indent=1)[:6000])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
