"""Question-answering over a company's own filings, with the citation enforced in code.

**The gap this closes.** Until now the console could say *that* a 10-Q was filed and could quote
one sentence of an earnings release (`market/earnings_release.py`), but it could not answer a
question whose answer sits inside a filing: "what does NVIDIA say drove data-center revenue",
"what did Coinbase disclose about its custody concentration", "how much did Apple return to
shareholders". A model asked that without the document invents a plausible answer; a model handed
the whole document and asked to "cite sources" writes citations that look right and point nowhere.
`lui/provenance.py` labels a line *after* an engine wrote it; nothing checked, at generation time,
that a sentence about a filing is backed by a specific passage of it.

**The contract, in four steps, each done by code rather than by asking the model nicely:**

1. **Every passage gets an id the model cannot guess.** A document is cut into overlapping chunks
   and each chunk gets ``dqa-`` plus eight hex characters of an MD5 over the document id and the
   chunk's first 500 characters. Taken from paper-qa: ``Context.populate_id``
   (`paperqa/types.py:279-316`, ``REFERENCE_TEMPLATE = "pqac-{id}"``, ``ID_HASH_LENGTH = 8``,
   ``CONTEXT_ENCODING_LENGTH = 500``) and ``encode_id`` (`paperqa/utils.py:244-250`). **Departure,
   deliberate:** paper-qa hashes the *question* into the id, so the same passage gets a new id for
   every question; ours hashes the *document*, so an id is stable across questions and a cached or
   logged answer still resolves tomorrow. **Why hash ids and not STORM's ``[1]..[n]``**
   (`knowledge_storm/.../grounded_question_answering.py:36`): a model that invents ``[3]`` when
   six passages were shown cites a real passage by accident, and nothing downstream can tell. An
   invented ``dqa-`` id resolves to nothing, so every fabricated citation is *detectable*.
2. **Retrieval trades relevance for diversity.** BM25 relevance, then Maximal Marginal Relevance
   over the ``fetch_k = 2k`` best: greedily add the chunk maximising
   ``lambda * relevance - (1 - lambda) * max_similarity_to_already_selected``. Taken from paper-qa
   ``VectorStore.max_marginal_relevance_search`` (`paperqa/llms.py:111-170`; the partitioned
   variant at `llms.py:89-109` is not needed because retrieval is already scoped to one ticker's
   filings). A 10-K repeats itself — the same revenue figure in the MD&A, the segment note and the
   selected-data table — and plain top-k spends the whole context on one fact said four ways.
   **Departure:** paper-qa embeds with a hosted embedding model; this uses BM25 for relevance and
   TF-IDF cosine for redundancy, both computed here with no dependency and no API call. Filings are
   dense with exact terms and figures where a lexical score is strong, and the metered Qwen key is
   not spent on embeddings. paper-qa ships ``texts_index_mmr_lambda = 1.0`` (`settings.py:804`),
   i.e. MMR *off* by default; the value used here, :data:`MMR_LAMBDA`, was chosen on the measured
   comparison in `eval/document_qa_eval.py`, not copied.
3. **The model is told to cite only those ids, after every sentence.** Prompt adapted from
   paper-qa ``qa_prompt`` and ``CITATION_KEY_CONSTRAINTS`` (`paperqa/prompts.py:36-67`) and STORM's
   ``AnswerQuestion`` signature ("make sure every sentence is supported by the gathered
   information", `grounded_question_answering.py:31-49`), plus paper-qa's explicit cannot-answer
   phrase (`prompts.py:28`) so an unanswerable question has a sanctioned exit.
4. **The answer is parsed back and enforced.** Citation ids are recovered with a tolerant regex, as
   paper-qa ``get_citation_ids`` does (`paperqa/utils.py:191-194`), so ``(a; b)`` and ``[a]`` still
   count. Then, sentence by sentence:

   * an id that resolves to no retrieved chunk is **removed** — paper-qa strips "leftover
     hallucinated citations" (`paperqa/types.py:510-515`);
   * a sentence left with **no** resolving citation is **dropped**. paper-qa does not do this: it
     strips the bad key and keeps the sentence, so a claim whose only citation was fabricated
     survives, uncited, in the rendered answer. Here it cannot;
   * optionally (on by default), a sentence stating a figure that appears in none of the chunks it
     cites is dropped — the model cited a real passage for a number that passage does not carry;
   * cited and merely-retrieved chunks are kept apart, as STORM keeps ``cited_info`` apart from
     ``raw_retrieved_info`` (`grounded_question_answering.py:151-163`,
     ``extract_cited_storm_info`` in `collaborative_storm_utils.py:86-105`). The console shows what
     the answer rests on, and separately what was read and not used.

   A trailing sentence cut off by the token limit is dropped, as STORM's
   ``remove_uncompleted_sentences_with_citations`` does (`knowledge_storm/utils.py:367-420`).

**Licences.** paper-qa (Future-House/paper-qa) is Apache-2.0 upstream — the local clone's LICENSE
file was edited to read MIT and is not relied on; see `licenses/paper_qa-APACHE-2.0.txt`. The
paper-qa pieces above are adapted, not vendored: the id scheme, the MMR loop and the prompt wording
were rewritten for this module (modified from the upstream files named). STORM
(stanford-oval/storm) is MIT.

**What was rejected, and why.** paper-qa's per-chunk "map" step (`core.py:383-401`), which asks the
model to summarise and score every retrieved chunk before answering: it costs one model call per
chunk — ten per question at its default ``evidence_k`` — on a key with a finite balance, and the
enforcement above does not depend on it. STORM's query-decomposition step (``QuestionToQuery``):
the corpus is a fixed set of filings, not a search engine, so there is nothing to search for.
paper-qa's 5,000-character chunk default (`settings.py:250`) is sized for the map step, which
reads chunks one at a time; without it twelve chunks go into one prompt, so chunks are smaller.

**Sources ARGUS reads.** SEC EDGAR primary documents (10-K, 10-Q, 8-K) and an 8-K's exhibit 99.1,
through the same keyless endpoints `market/evidence.EdgarSource` and
`market/earnings_release.py` already use. News arrives in ARGUS as RSS headlines with no body
(`market/evidence.Headline` carries title and link only); :func:`headline_documents` turns them
into one-chunk documents so a headline can be cited, but a headline is not an article and no
article body is fetched. That is stated rather than implied.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from argus.llm.base import ChatModel
from argus.llm.qwen import Thinking
from argus.lui.answer import Source
from argus.market.evidence import EdgarSource

# --- fetching ---------------------------------------------------------------------------------

_UA = "ARGUS research desk research@argus-desk.example"
"""The SEC blocks anonymous clients; the same descriptive agent `market/earnings_release.py`
uses."""
_TIMEOUT = 30.0
_MAX_BYTES = 12_000_000
"""A 10-K's primary HTML runs 1.5-6 MB with inline XBRL. Cap the read so one pathological filing
cannot hold the console."""
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/"

FORMS = ("10-K", "10-Q", "8-K")
"""The filings this reads. A 10-K/10-Q is the periodic statement; an 8-K is the event."""


@dataclass(frozen=True)
class Document:
    """One document the answerer may cite, with enough to send a reader to it."""

    doc_id: str
    """``{ticker}-{form}-{accession}`` for a filing (``-ex99`` for an exhibit)."""
    ticker: str
    form: str
    """``10-K`` | ``10-Q`` | ``8-K`` | ``EX-99.1`` | ``headline``."""
    filed: date
    url: str
    title: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def label(self) -> str:
        return f"{self.ticker} {self.form} filed {self.filed:%d %b %Y}"


_BLOCK_END = re.compile(
    r"</(?:p|div|tr|li|h[1-6]|table|section|title)\s*>|<br\s*/?>", re.I)
_CELL_END = re.compile(r"</t[dh]\s*>", re.I)
_HIDDEN = re.compile(
    r"<(script|style|ix:header|head)\b.*?</\1\s*>|<!--.*?-->", re.I | re.S)
_DISPLAY_NONE = re.compile(
    r"<div[^>]*display:\s*none[^>]*>.*?</div>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(document: str) -> str:
    """Plain text that keeps the document's block structure.

    `market/earnings_release.clean` collapses every run of whitespace to one space, which is right
    for sentence extraction and wrong here: without line breaks a chunker cannot cut on paragraph
    or table-row boundaries, and a financial table becomes one line of numbers with no row labels.
    So block ends become newlines, table cells become ``" | "``, and the inline-XBRL header — tens
    of kilobytes of machine facts that are not prose — is removed with scripts and styles.
    """
    text = _HIDDEN.sub(" ", document)
    text = _DISPLAY_NONE.sub(" ", text)
    text = _CELL_END.sub(" | ", text)
    text = _BLOCK_END.sub("\n", text)
    text = html.unescape(_TAG.sub(" ", text)).replace("\xa0", " ")
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        line = re.sub(r"(?:\|\s*){2,}", "| ", line).strip(" |")
        # A lone ``$``, ``)`` or ``%`` cell is table furniture, not content.
        if len(line) > 1 and re.search(r"[A-Za-z0-9]", line):
            lines.append(line)
    return "\n".join(lines)


def _fetch(url: str) -> str:
    # Fixed https host (sec.gov); the path comes from EDGAR's own index, not from a user.
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return str(response.read(_MAX_BYTES).decode("utf-8", errors="replace"))


class EdgarDocuments:
    """The latest 10-K, 10-Q and 8-K of a ticker, read in full from EDGAR.

    Built on `market/evidence.EdgarSource` for the ticker → CIK map and the submissions index, so
    the SEC's rate limit and user-agent rule are honoured in one place. The primary document path
    is EDGAR's own ``primaryDocument`` field; the exhibit lookup reads the filing's index page the
    way `market/earnings_release.EarningsReleaseSource._exhibit_url` does.
    """

    def __init__(self, edgar: EdgarSource | None = None) -> None:
        self._edgar = edgar or EdgarSource(user_agent=_UA)

    def _exhibit(self, cik: int, accession: str, prefix: str) -> str | None:
        folder = accession.replace("-", "")
        index = _fetch(ARCHIVE.format(cik=cik, folder=folder) + f"{accession}-index.htm")
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S):
            cells = [re.sub(r"<[^>]+>", "", c).strip()
                     for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            links = re.findall(r'href="([^"]+\.htm[l]?)"', row)
            if any(c.upper().startswith(prefix) for c in cells) and links:
                href = links[0]
                return f"https://www.sec.gov{href}" if href.startswith("/") else href
        return None

    def latest(self, ticker: str, forms: Sequence[str] = FORMS, *,
               with_exhibit: bool = True) -> list[Document]:
        """The newest filing of each form in ``forms``; for an 8-K, its EX-99.1 too."""
        cik = self._edgar.cik_for(ticker)
        if cik is None:
            return []
        recent = self._edgar._get(self._edgar.SUBMISSIONS_URL.format(cik=cik))
        rf = recent.get("filings", {}).get("recent", {})
        wanted = set(forms)
        out: list[Document] = []
        for i, form in enumerate(rf.get("form", [])):
            if form not in wanted:
                continue
            wanted.discard(form)
            accession = rf["accessionNumber"][i]
            folder = accession.replace("-", "")
            url = ARCHIVE.format(cik=cik, folder=folder) + rf["primaryDocument"][i]
            filed = date.fromisoformat(rf["filingDate"][i])
            items = (rf.get("items", [""] * (i + 1))[i] or "").strip()
            title = f"{ticker.upper()} {form}" + (f" (items {items})" if items else "")
            out.append(Document(doc_id=f"{ticker.upper()}-{form}-{accession}",
                                ticker=ticker.upper(), form=form, filed=filed, url=url,
                                title=title, text=html_to_text(_fetch(url))))
            if form == "8-K" and with_exhibit:
                exhibit = self._exhibit(cik, accession, "EX-99.1")
                if exhibit is not None:
                    out.append(Document(
                        doc_id=f"{ticker.upper()}-8-K-{accession}-ex99", ticker=ticker.upper(),
                        form="EX-99.1", filed=filed, url=exhibit,
                        title=f"{ticker.upper()} 8-K exhibit 99.1",
                        text=html_to_text(_fetch(exhibit))))
            if not wanted:
                break
        return out


def headline_documents(ticker: str, headlines: Iterable[Any]) -> list[Document]:
    """Headlines (`market/evidence.Headline`: ``feed``, ``title``, ``link``, ``published``) as
    one-chunk documents. A headline is all ARGUS holds of a news item; no body is fetched."""
    out: list[Document] = []
    for h in headlines:
        key = hashlib.md5(f"{h.link}{h.title}".encode()).hexdigest()[:10]
        out.append(Document(doc_id=f"{ticker.upper()}-headline-{key}", ticker=ticker.upper(),
                            form="headline", filed=h.published.date(), url=h.link,
                            title=f"{h.feed} headline", text=h.title))
    return out


# --- chunking ---------------------------------------------------------------------------------

CHUNK_CHARS = 1500
"""Characters per chunk. :data:`TOP_K` of these plus instructions is about 4.5k prompt tokens,
which leaves the answer room inside a LOW-thinking call. paper-qa's 5,000 is sized for its map
step; see the module docstring."""
CHUNK_OVERLAP = 200
"""Carried from the end of one chunk into the next, so a sentence split at a boundary is whole in
at least one of them. paper-qa's default is 250 on 5,000 (`settings.py:250`); ours is a larger
share because our chunks are smaller."""
ID_TEMPLATE = "dqa-{id}"
ID_HASH_LENGTH = 8
CONTEXT_ENCODING_LENGTH = 500
_ID = re.compile(r"\bdqa-[0-9a-f]{8}\b")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: Document
    index: int
    text: str


def _encode_id(value: str) -> str:
    """paper-qa ``encode_id`` (`paperqa/utils.py:244-250`): lower-cased MD5, truncated."""
    return hashlib.md5(value.lower().encode()).hexdigest()[:ID_HASH_LENGTH]


def _split_long(line: str, limit: int) -> list[str]:
    """A line longer than a chunk, cut at sentence ends where possible, else hard."""
    parts: list[str] = []
    rest = line
    while len(rest) > limit:
        cut = max(rest.rfind(". ", 0, limit), rest.rfind("; ", 0, limit))
        cut = cut + 1 if cut > limit // 3 else limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def chunk(doc: Document, *, chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Cut on line boundaries into chunks of at most ``chars``, carrying ``overlap`` forward.

    paper-qa's ``chunk_pdf`` (`paperqa/readers.py:105-142`) cuts at a fixed character offset,
    which splits words and table rows; cutting on the block boundaries :func:`html_to_text` keeps
    means a table row is never halved.
    """
    lines: list[str] = []
    for line in doc.text.split("\n"):
        lines.extend(_split_long(line, chars))
    chunks: list[Chunk] = []
    current: list[str] = []
    size = 0
    seen: set[str] = set()

    def emit() -> None:
        text = "\n".join(current)
        base = ID_TEMPLATE.format(id=_encode_id(doc.doc_id + text[:CONTEXT_ENCODING_LENGTH]))
        cid, salt = base, 0
        while cid in seen:  # two chunks with the same opening: disambiguate, never collide
            salt += 1
            cid = ID_TEMPLATE.format(id=_encode_id(f"{doc.doc_id}{salt}{text}"))
        seen.add(cid)
        chunks.append(Chunk(id=cid, doc=doc, index=len(chunks), text=text))

    for line in lines:
        if current and size + len(line) + 1 > chars:
            emit()
            carried: list[str] = []
            kept = 0
            for prior in reversed(current):
                if kept + len(prior) + 1 > overlap:
                    break
                carried.insert(0, prior)
                kept += len(prior) + 1
            current, size = carried, kept
        current.append(line)
        size += len(line) + 1
    if current and (not chunks or size > overlap):
        emit()
    return chunks


# --- retrieval --------------------------------------------------------------------------------

MMR_LAMBDA = 0.85
"""Weight on relevance against redundancy; 1.0 is plain top-k. Measured 2026-09-25 in
`eval/document_qa_eval.py` on the 27 answerable questions over the frozen corpus (does the retrieved
set contain the passage that states the answer): at k=12, lambda 1.0 found 25, **0.85 found 26**,
0.7 found 25; at k=8, 23 / 24 / 23; at 0.5 recall fell at every k. Near-duplicate pairs among the
retrieved set fell by about a third from 1.0 to 0.85. The gain is one question in 27 and was
chosen on the same questions it is reported on, so it is a small, in-sample edge, not a
generalisation claim; what MMR reliably buys is less redundancy for no loss of recall."""
TOP_K = 12
"""Chunks per prompt. k=12 recalled the answer passage for 26 of 27 questions, k=8 for 24 (same
measurement). Twelve 1,500-character chunks is about 4.5k prompt tokens."""
BM25_K1 = 1.5
BM25_B = 0.75

_STOP = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "what", "when", "which", "who", "why", "with", "company", "companys",
    "say", "says", "said", "report", "reported", "disclose", "disclosed", "according",
))
_TOKEN = re.compile(r"[a-z][a-z0-9\-]+|\d+(?:\.\d+)?")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower().replace(",", "").replace("'", ""))
            if t not in _STOP]


class Index:
    """BM25 relevance and TF-IDF cosine redundancy over one set of chunks."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self._tf = [Counter(tokens(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        n = len(self.chunks)
        self._avg = (sum(self._len) / n) if n else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._norm = [math.sqrt(sum((v * self._idf[t]) ** 2 for t, v in tf.items())) or 1.0
                      for tf in self._tf]

    def bm25(self, query: str) -> list[float]:
        q = set(tokens(query))
        scores: list[float] = []
        for tf, length in zip(self._tf, self._len, strict=True):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (BM25_K1 + 1) / (
                        f + BM25_K1 * (1 - BM25_B + BM25_B * length / (self._avg or 1.0)))
            scores.append(s)
        return scores

    def cosine(self, i: int, j: int) -> float:
        a, b = self._tf[i], self._tf[j]
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(t, 0) * self._idf[t] ** 2 for t, v in a.items())
        return dot / (self._norm[i] * self._norm[j])

    def search(self, query: str, *, k: int = TOP_K, mmr_lambda: float = MMR_LAMBDA,
               fetch_k: int | None = None) -> list[tuple[Chunk, float]]:
        """Top ``k`` by MMR over the ``fetch_k`` (default ``2k``) best by BM25.

        The greedy loop is paper-qa's (`paperqa/llms.py:141-170`): seed with the most relevant,
        then repeatedly take the argmax of ``lambda * rel - (1 - lambda) * max_sim_to_selected``.
        Relevance is BM25 divided by the best BM25 score, so it lives on the same [0, 1] scale as
        the cosine it is traded against — paper-qa's cosine scores already do.
        """
        fetch = fetch_k if fetch_k is not None else 2 * k
        if fetch < k:
            raise ValueError("fetch_k must be greater or equal to k")
        raw = self.bm25(query)
        order = sorted((i for i, s in enumerate(raw) if s > 0), key=lambda i: -raw[i])[:fetch]
        if not order:
            return []
        top = raw[order[0]]
        rel = {i: raw[i] / top for i in order}
        if len(order) <= k or mmr_lambda >= 1.0:
            return [(self.chunks[i], raw[i]) for i in order[:k]]
        selected = [order[0]]
        remaining = order[1:]
        while len(selected) < k and remaining:
            best = max(remaining, key=lambda i: mmr_lambda * rel[i] - (1 - mmr_lambda)
                       * max(self.cosine(i, j) for j in selected))
            selected.append(best)
            remaining.remove(best)
        return [(self.chunks[i], raw[i]) for i in selected]


def redundancy(index: Index, picked: Sequence[Chunk], threshold: float = 0.6) -> float:
    """Share of retrieved pairs that are near-duplicates (cosine at or above ``threshold``)."""
    pos = {c.id: n for n, c in enumerate(index.chunks)}
    ids = [pos[c.id] for c in picked]
    pairs = [(a, b) for n, a in enumerate(ids) for b in ids[n + 1:]]
    if not pairs:
        return 0.0
    return sum(index.cosine(a, b) >= threshold for a, b in pairs) / len(pairs)


# --- prompting --------------------------------------------------------------------------------

MAX_TOKENS = 2000
"""Completion ceiling for the answer call. Reasoning counts against it — about 650 tokens at LOW
thinking (`llm/qwen.Thinking`) — so this leaves roughly 1,300 for six sentences with keys."""

CANNOT_ANSWER = "I cannot answer"
"""paper-qa's ``CANNOT_ANSWER_PHRASE`` (`paperqa/prompts.py:28`). A sanctioned way out, so a
question the filings do not answer is refused rather than answered from memory."""

CITATION_RULES = (
    "Valid citations, one parenthetical of comma-separated keys at the end of the sentence:\n"
    "- (dqa-d79ef6fa, dqa-0f650d59)\n"
    "- (dqa-d79ef6fa)\n"
    "Invalid: (dqa-d79ef6fa and dqa-0f650d59), (dqa-d79ef6fa;dqa-0f650d59), a key you were not "
    "given, a page number, an author, a year."
)
"""Adapted from paper-qa ``CITATION_KEY_CONSTRAINTS`` (`paperqa/prompts.py:36-46`)."""

SYSTEM = (
    "You answer questions about a company using only the numbered filing passages provided. "
    "You never use outside knowledge, never estimate, and never compute a figure the passages "
    "do not state."
)


def build_prompt(question: str, retrieved: Sequence[Chunk]) -> list[dict[str, Any]]:
    """The answer prompt. Adapted from paper-qa ``qa_prompt`` (`paperqa/prompts.py:48-67`) and
    STORM ``AnswerQuestion`` (`grounded_question_answering.py:31-49`)."""
    blocks = "\n\n".join(
        f"[{c.id}] {c.doc.label} — {c.doc.title}\n{c.text}" for c in retrieved)
    user = (
        f"Passages:\n\n{blocks}\n\n---\n\nQuestion: {question}\n\n"
        "Answer in at most six short sentences, plain prose, no lists and no headings. Every "
        "sentence must be supported by the passages and must end with the key(s) of the "
        "passage(s) that support it. State figures exactly as the passages state them, with "
        "their units and periods. If the passages do not contain the answer, reply exactly "
        f'"{CANNOT_ANSWER}." and nothing else.\n\n{CITATION_RULES}'
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# --- enforcement ------------------------------------------------------------------------------

_ABBREV = re.compile(
    r"(?:\b(?:U\.S|Inc|Corp|Co|Ltd|No|Nos|vs|approx|e\.g|i\.e|etc|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
    r"Sep|Sept|Oct|Nov|Dec|Mr|Ms|Dr|St)\.|\b[A-Z]\.)$")
_CITE_ONLY = re.compile(r"^[\s(\[]*(?:dqa-[0-9a-f]{8}[\s,;)\]]*)+[.]?$")
_PAREN_CITES = re.compile(r"\s*[(\[](?:[^()\[\]]*?dqa-[0-9a-f]{8}[^()\[\]]*?)[)\]]")
_LEADING_CITES = re.compile(r"^\s*((?:[(\[][^()\[\]]*?dqa-[0-9a-f]{8}[^()\[\]]*?[)\]]\s*)+)")
_BARE_ID = re.compile(r"\s*\bdqa-[0-9a-z]{1,16}\b")


def split_sentences(text: str) -> list[str]:
    """Sentences, with a citation-only fragment folded into the sentence before it.

    A split after ``.``/``!``/``?`` and whitespace is refused when the text before it ends in a
    known abbreviation (``U.S.``, ``Inc.``) — a false split would leave a fragment with no
    citation, which enforcement would then drop, silently deleting half a true sentence.
    """
    pieces: list[str] = []
    for para in re.split(r"\n+", text.strip()):
        para = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", para).strip()
        if not para:
            continue
        start = 0
        for m in re.finditer(r"(?<=[.!?])[)\]]?\s+", para):
            before = para[start:m.start()]
            if _ABBREV.search(before.rstrip(")]")):
                continue
            if m.end() < len(para) and para[m.end()].islower():
                continue
            pieces.append(para[start:m.end()].strip())
            start = m.end()
        if para[start:].strip():
            pieces.append(para[start:].strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and _CITE_ONLY.match(piece):
            merged[-1] = f"{merged[-1]} {piece}"
            continue
        # "Revenue was $89.0 billion. (dqa-5908b085) It rose 117%. (dqa-5908b085)" — the live
        # model put the key after the full stop on 2026-09-25. Split on the stop, the key opens
        # the *next* piece and would be credited to the wrong sentence, leaving the one it
        # belongs to uncited. A key group that opens a piece belongs to the sentence before it.
        lead = _LEADING_CITES.match(piece)
        if merged and lead:
            merged[-1] = f"{merged[-1]} {lead.group(1).strip()}"
            piece = piece[lead.end():].strip()
            if not piece:
                continue
        merged.append(piece)
    return merged


def citation_ids(text: str) -> list[str]:
    """Every ``dqa-`` key in order of first appearance. paper-qa ``get_citation_ids``
    (`paperqa/utils.py:191-194`)."""
    return list(dict.fromkeys(_ID.findall(text)))


def strip_citations(sentence: str) -> str:
    text = _PAREN_CITES.sub("", sentence)
    text = _BARE_ID.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# --- figures ----------------------------------------------------------------------------------

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
          "k": 1e3, "m": 1e6, "mn": 1e6, "bn": 1e9, "b": 1e9}
_FIGURE = re.compile(
    r"(?P<cur>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b|per cent\b)"
    r"|\s?(?P<scale>thousand|million|billion|trillion|bn|mn)\b)?", re.I)
_YEARISH = re.compile(r"^(?:19|20)\d{2}$")


@dataclass(frozen=True)
class Figure:
    text: str
    value: float
    """In units (a percentage as its number, 56% -> 56)."""
    percent: bool
    step: float
    """The precision the figure was written to, in the same units as ``value``: ``$46.7 billion``
    is written to the nearest 0.1 billion, so ``step`` is 1e8 and anything that rounds to it
    matches."""


def figures(text: str) -> list[Figure]:
    """Figures a claim states: amounts, percentages and multi-digit numbers, not years or dates.

    A small bare integer ("three segments", "Q2", "item 7") is not a figure worth checking, and a
    four-digit year is a date. Everything with a ``$``, a ``%``, a scale word, a decimal point or a
    thousands separator is.
    """
    text = re.sub(r"\bdqa-[0-9a-f]{8}\b", " ", text)
    out: list[Figure] = []
    for m in _FIGURE.finditer(text):
        num = m.group("num")
        cur, pct, scale = m.group("cur"), m.group("pct"), m.group("scale")
        # The match can open on the optional space before the number, so look behind the
        # figure itself, not behind the match: "to 76.0%" must not read as "o76.0%".
        lead = m.start("cur") if cur else m.start("num")
        before = text[max(0, lead - 1):lead]
        if before.isalpha() or (before == "-" and lead > 1 and text[lead - 2].isalpha()):
            continue  # part of an identifier: "Q2", "10-K", "H100"
        plain = num.replace(",", "")
        bare = not (cur or pct or scale)
        if bare and (_YEARISH.match(plain) or (len(plain) < 3 and "." not in num)):
            continue
        value = float(plain)
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        mult = _SCALE.get((scale or "").lower(), 1.0)
        out.append(Figure(text=m.group(0).strip(), value=value * mult, percent=bool(pct),
                          step=10.0 ** (-decimals) * mult))
    return out


def _numbers_in(text: str) -> list[float]:
    values: list[float] = []
    for m in re.finditer(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", text):
        values.append(float(m.group(0).replace(",", "")))
    return values


def figure_supported(fig: Figure, passage: str) -> bool:
    """Does ``passage`` carry this figure, at any reporting scale and to the claim's precision?

    Filings print tables in thousands or millions ("$ in millions"), and a correct sentence says
    "$46.7 billion" for a cell that reads ``46,743``. So a passage number ``x`` matches when
    ``x * s`` for ``s`` in units/thousands/millions/billions rounds to the claim at the precision
    the claim was written to. A percentage matches only a passage number, at the same rounding —
    a claim of 56% is carried by "56%" or "55.8%", and not by an unrelated "$56 million".
    """
    step = fig.step
    for x in _numbers_in(passage):
        scales = (1.0,) if fig.percent else (1.0, 1e3, 1e6, 1e9)
        for s in scales:
            if abs(x * s - fig.value) <= step / 2 + 1e-9 * max(1.0, abs(fig.value)):
                return True
    return False


# --- the answer -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dropped:
    sentence: str
    reason: str
    """``uncited`` | ``unresolved_citation`` | ``figure_not_in_citation`` | ``truncated``."""
    ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    text: str
    cited: tuple[str, ...]


@dataclass
class DocumentAnswer:
    """One answer. ``claims`` survived enforcement; ``dropped`` did not, and says why."""

    question: str
    claims: list[Claim] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    retrieved: list[Chunk] = field(default_factory=list)
    hallucinated_ids: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    raw: str = ""

    @property
    def cited_ids(self) -> list[str]:
        return list(dict.fromkeys(i for c in self.claims for i in c.cited))

    def _chunk(self, cid: str) -> Chunk:
        return next(c for c in self.retrieved if c.id == cid)

    @property
    def cited(self) -> list[Chunk]:
        return [self._chunk(i) for i in self.cited_ids]

    @property
    def merely_retrieved(self) -> list[Chunk]:
        used = set(self.cited_ids)
        return [c for c in self.retrieved if c.id not in used]

    @property
    def lines(self) -> list[str]:
        """What a reader sees: each claim with ``[n]`` markers into :attr:`sources`."""
        if self.refused:
            return [self.reason]
        number = {cid: n + 1 for n, cid in enumerate(self.cited_ids)}
        return [c.text + "".join(f"[{number[i]}]" for i in c.cited) for c in self.claims]

    @property
    def sources(self) -> list[Source]:
        """Cited sources, in the order the ``[n]`` markers number them."""
        return [_source(c, n + 1, cited=True) for n, c in enumerate(self.cited)]

    @property
    def retrieved_sources(self) -> list[Source]:
        """Read and not cited — shown apart, never mixed into :attr:`sources`."""
        return [_source(c, 0, cited=False) for c in self.merely_retrieved]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "refused": self.refused,
            "reason": self.reason,
            "lines": self.lines,
            "claims": [{"text": c.text, "cited": list(c.cited)} for c in self.claims],
            "dropped": [{"sentence": d.sentence, "reason": d.reason, "ids": list(d.ids)}
                        for d in self.dropped],
            "hallucinated_ids": list(self.hallucinated_ids),
            "cited": [c.id for c in self.cited],
            "merely_retrieved": [c.id for c in self.merely_retrieved],
            "sources": [s.as_dict() for s in self.sources],
            "retrieved_sources": [s.as_dict() for s in self.retrieved_sources],
        }


def _source(c: Chunk, n: int, *, cited: bool) -> Source:
    snippet = re.sub(r"\s+", " ", c.text)[:160]
    marker = f"[{n}] " if cited else "retrieved, not cited · "
    return Source(kind="evidence", ref=f"{c.doc.url}#{c.id}",
                  detail=f"{marker}{c.doc.label} · {c.id} · “{snippet}…”")


def enforce(question: str, raw: str, retrieved: Sequence[Chunk], *,
            require_figures: bool = True, truncated: bool = False) -> DocumentAnswer:
    """Parse a model's answer and keep only what its citations carry. See the module docstring."""
    by_id = {c.id: c for c in retrieved}
    out = DocumentAnswer(question=question, retrieved=list(retrieved), raw=raw)
    text = raw.strip()
    if not text or CANNOT_ANSWER.lower() in text.lower()[:len(CANNOT_ANSWER) + 40]:
        out.refused = True
        out.reason = ("The filings read do not answer this."
                      if text else "The model returned nothing.")
        return out
    sentences = split_sentences(text)
    if truncated and sentences and not re.search(r"[.!?][)\]]?\s*$", sentences[-1]):
        out.dropped.append(Dropped(sentences.pop(), "truncated"))
    for sentence in sentences:
        ids = citation_ids(sentence)
        bad = [i for i in ids if i not in by_id]
        good = tuple(i for i in ids if i in by_id)
        out.hallucinated_ids.extend(i for i in bad if i not in out.hallucinated_ids)
        # Anything shaped like a key but not one ("dqa-12" or an upper-cased variant) is a
        # fabricated citation too.
        for fake in re.findall(r"\bdqa-[0-9A-Za-z]{1,16}\b", sentence):
            if not _ID.fullmatch(fake) and fake not in out.hallucinated_ids:
                out.hallucinated_ids.append(fake)
        body = strip_citations(sentence)
        if not re.search(r"[A-Za-z0-9]", body):
            continue
        if not good:
            reason = "unresolved_citation" if (bad or "dqa-" in sentence) else "uncited"
            out.dropped.append(Dropped(body, reason, tuple(bad)))
            continue
        if require_figures:
            cited_text = "\n".join(by_id[i].text for i in good)
            missing = [f for f in figures(body) if not figure_supported(f, cited_text)]
            if missing:
                out.dropped.append(Dropped(body, "figure_not_in_citation", good))
                continue
        out.claims.append(Claim(body, good))
    if not out.claims:
        out.refused = True
        out.reason = ("No sentence of the model's answer was backed by a passage it cited, so "
                      "nothing is shown.")
    return out


def answer(question: str, chunks: Sequence[Chunk], model: ChatModel, *, k: int = TOP_K,
           mmr_lambda: float = MMR_LAMBDA, require_figures: bool = True,
           max_tokens: int = MAX_TOKENS, index: Index | None = None) -> DocumentAnswer:
    """Retrieve, ask, enforce. One model call."""
    idx = index or Index(chunks)
    retrieved = [c for c, _ in idx.search(question, k=k, mmr_lambda=mmr_lambda)]
    if not retrieved:
        return DocumentAnswer(question=question, refused=True,
                              reason="Nothing in the filings read matches the question.")
    completion = model.complete(build_prompt(question, retrieved), temperature=0.0,
                                max_tokens=max_tokens, thinking=Thinking.LOW)
    return enforce(question, completion.content, retrieved, require_figures=require_figures,
                   truncated=completion.finish_reason == "length")


def research_answer(question: str, ticker: str, model: ChatModel, *,
                    documents: Sequence[Document] | None = None,
                    source: EdgarDocuments | None = None) -> tuple[list[str], list[Source],
                                                                   dict[str, Any]]:
    """The console's call: ``(lines, sources, data)`` for `lui/research.py`.

    Reads the ticker's latest 10-K, 10-Q and 8-K (with exhibit 99.1) unless ``documents`` is
    given. ``sources`` holds only cited passages; the read-but-uncited ones are in
    ``data["retrieved_sources"]`` so the console can show them apart.
    """
    docs = list(documents) if documents is not None else (
        source or EdgarDocuments()).latest(ticker)
    if not docs:
        return ([f"No 10-K, 10-Q or 8-K could be read for {ticker.upper()} on EDGAR."], [],
                {"refused": True})
    chunks = [c for d in docs for c in chunk(d)]
    result = answer(question, chunks, model)
    header = ("From " + ", ".join(d.label for d in docs if d.form != "headline")
              + " — each sentence below is backed by the passage it cites; sentences that were "
                "not were removed.")
    lines = result.lines if result.refused else [header, *result.lines]
    return lines, result.sources, result.as_dict()
