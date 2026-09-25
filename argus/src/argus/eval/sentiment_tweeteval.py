"""ARGUS's sentiment reading on TweetEval, with TweetEval's own metric — and the crowd read's abuse
screen checked against TweetEval's hate and offensive test sets.

**Two questions, one benchmark.** `eval/sentiment_comparison.py` measured the axis ARGUS claims —
repetition does not make its sentiment analyst louder — and said in so many words that it did not
race finBERT on classification. This module runs the race it declined, on a public, labelled,
leaderboard-comparable test set, so the disclaimed axis is a number rather than a sentence:
TweetEval's sentiment task (Barbieri et al., Findings of EMNLP 2020, arXiv:2010.12421v2; the data
is SemEval-2017 Task 4 subtask A), 12,284 test tweets, three classes. The second question is a
regression check on `market/social_pulse.py`: TweetEval's hate (SemEval-2019 HatEval, 2,970 test
tweets) and offensive (SemEval-2019 OffensEval, 860) sets are run through the real ``pulse()`` path
to see whether abusive text is counted as crowd signal or quoted back to a reader.

**Why macro-averaged recall, not accuracy or F1.** TweetEval scores sentiment by macro-averaged
recall (paper §2.2, "Evaluation metrics": "the same evaluation metric from the original tasks …
sentiment analysis (macro-averaged recall)"), and the original task says why (Rosenthal, Farra and
Nakov, SemEval-2017 Task 4, §4.1): recall averaged over the three classes is 1/3 for *any* trivial
classifier that assigns every tweet one class, and also the expected value of a random one, while
the accuracy of the majority-class classifier equals the majority class's prevalence, "which may
be much higher than 0.5 if the test set is imbalanced"; F1 is "sensitive to class imbalance for
the same reason", and unlike F1 macro-recall does not change if positive and negative are swapped.
This test set is imbalanced — 48.3% neutral, 32.3% negative, 19.3% positive — so an always-neutral
scorer gets 48.3% accuracy and 1/3 macro-recall. That is not an abstract point here: ARGUS's crowd
read *is* an always-neutral scorer (below), and accuracy would have flattered it by 15 points.
Hate and offensive are scored by macro-F1, TweetEval's metric for both; the regression check adds
the two numbers the pulse actually operates on — the share of abusive posts withheld (recall on
the abusive class) and the share of clean posts withheld by mistake.

**What is compared, on the same test split.**

- *ARGUS's crowd read* (`market/social_pulse.pulse`), run on the tweets in batches of 60 — the
  size of one name's sweep. Until 2026-09-26 it assigned no polarity and scored the floor, 1/3.
  It now scores every post the abuse screen keeps with VADER, and each tweet is given the class
  the pulse itself published for that post id (``post_tone``) — not a class recomputed here. A
  post the screen withholds gets no tone from the pulse, so it is scored as neutral, which is
  what a reader of the console is told about it (nothing). The same tweets are also run with the
  screen off, which must reproduce VADER alone exactly; the difference between the two runs is the
  screen's cost, measured per tweet (macro-recall, paired bootstrap) and per sweep (the tone
  shares a reader sees, against the gold shares of the whole sweep). If the pulse stops scoring a
  kept post, this module raises rather than scoring it as neutral.
- *ARGUS's Sentiment Integrity Analyst* (`agents/analysts.SentimentAnalyst`) — **not run**. It
  needs Qwen and this item's Qwen budget is zero; the artefact states the sample size a run would
  need, derived from the measured bootstrap spread, rather than a number nobody measured.
- *finBERT* (``ProsusAI/finbert``, the "Market sentiment" capability's named baseline), from the
  local Hugging Face cache through `eval/baselines/finbert_loader.py`, offline, every test tweet.
- *VADER* (``cjhutto/vaderSentiment``, MIT), the social-media lexicon scorer, from the local clone
  through `eval/baselines/vader_loader.py`, with its README's ±0.05 thresholds.
- *The majority class*, computed, and *the paper's baselines*, cited from Table 3 (test block) —
  SVM, FastText, BLSTM and three RoBERTa variants — with BERTweet and TimeLMs-2021 from the
  repository README's leaderboard. Where the README disagrees with the paper (RoBERTa-Retrained
  sentiment 72.8 vs Table 3's 72.6; RoBERTa-Twitter hate 49.9 vs 46.9), the paper's value is used
  and the README's is recorded beside it.
- *RoB-RT, re-scored.* The repository ships the paper's best model's test predictions — labels
  only, no text (:data:`REFERENCE_BLOBS`) — so the strongest cited baseline is also *run through
  this module's metric on the same items*, on all three tasks, with paired bootstrap differences
  against ARGUS's rows. That makes the reimplemented metric checkable against the paper too: it
  re-scores RoB-RT's sentiment predictions at 0.729 against Table 3's 72.6 (a mean of three runs)
  and the README's 72.8 (measured 2026-09-25).

**How the abuse screen was chosen (`market/abuse.py`).** Candidate configurations were compared
on TweetEval's *validation* splits only, and the one kept is the one that withholds the largest
share of abusive posts — mean of hate-class and offensive-class recall — while withholding at most
6% of clean posts on OffensEval validation and at most 6% of ordinary tweets on the sentiment
validation split. That criterion was fixed after a first validation pass, not before; to keep the
choice from flattering the result, every candidate's *test* numbers are published too, not only
the chosen one's. Test sets were read once the configuration was fixed. The market-jargon
allowlist is not a candidate: TweetEval contains almost none of the text it exists for, so its
effect is reported separately, on the benchmark and on the desk's own X and Reddit snapshot
(:data:`WITHOUT_ALLOWLIST`).

**Licence and data handling.** TweetEval publishes no licence (``gh api
repos/cardiffnlp/tweeteval/license`` answers 404; its README says restrictions of the underlying
datasets and of Twitter apply; the ``LICENSE`` in the local copy is not upstream). So nothing from
it is copied: its data is read in place from the research corpus, the metrics are reimplemented
here from the two papers' definitions (no line of ``evaluation_script.py`` is used — it is five
calls into scikit-learn, and the per-task choice of metric it encodes at lines 66-93 is taken as
behaviour), and ``data/sentiment_tweeteval.json`` holds counts and scores only.
:func:`assert_no_text` enforces that last rule on the artefact before it is written. Every file
read is identified by its git blob hash and checked against the upstream tree at commit 4fbd22c.

No Qwen call is made anywhere in this module.

    python -m argus.eval.sentiment_tweeteval
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from argus.market import abuse, social_pulse
from argus.market.abuse import DEFAULT_SCREEN, AbuseScreen

DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "sentiment_tweeteval.json"
DEFAULT_TWEETEVAL = (Path(__file__).resolve().parents[5] / "mypr" / "08_sentiment_and_events"
                     / "tweeteval" / "datasets")
"""The research corpus's copy. ``ARGUS_TWEETEVAL_DIR`` or ``--tweeteval`` points elsewhere."""

UPSTREAM = "https://github.com/cardiffnlp/tweeteval"
UPSTREAM_COMMIT = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
UPSTREAM_BLOBS: dict[str, str] = {
    "hate/mapping.txt": "44b259c804f1e8dd9c5e25ddfe980356dc0ccffd",
    "hate/test_labels.txt": "d9d3a798964e229839afae91d7e0f0fef17b5dad",
    "hate/test_text.txt": "a0055d380adc200e324f03da6435400245134c88",
    "hate/val_labels.txt": "421514e2a76742f1c45ee0a98ceeb417731e0aeb",
    "hate/val_text.txt": "85d1032af60d3389604da2b6c40d0b3f3ffb22f3",
    "offensive/mapping.txt": "1019138231fb5b942f8b360f8aa8c155f823d833",
    "offensive/test_labels.txt": "07a2b53e7cfa17a3c95e921e57c2dd81045a429d",
    "offensive/test_text.txt": "0c33e87ec0046f7268725382b4609cc4d308b666",
    "offensive/val_labels.txt": "619bc73701379cf4eaf3e526049aa5bb6a1a301a",
    "offensive/val_text.txt": "47355787a1ef2ecc4c3bf275c12c004e0803b4c6",
    "sentiment/mapping.txt": "e98272f57db16af409fee64d9efccf5116e1ec27",
    "sentiment/test_labels.txt": "e3cf7395a85ad38caf6b53cfdeeefd7fbf2dcc48",
    "sentiment/test_text.txt": "95fe0b5fdfaba658fca9e4d85ab9e5224bc67bbe",
    "sentiment/val_labels.txt": "fd86539deb0d094782a1efed6dd06416fac75566",
    "sentiment/val_text.txt": "b608664639361a9f8da99a882472af44f0330c89",
}
"""Git blob hashes of every file this module reads, from the upstream tree at
:data:`UPSTREAM_COMMIT` (``gh api repos/cardiffnlp/tweeteval/git/trees/<commit>?recursive=1``,
2026-09-25)."""

PAPER = ("Barbieri, Camacho-Collados, Espinosa-Anke, Neves. TweetEval: Unified Benchmark and "
         "Comparative Evaluation for Tweet Classification. Findings of EMNLP 2020. "
         "arXiv:2010.12421v2, Table 3 (test block; neural models are the mean of three runs).")
PAPER_TABLE3_TEST: dict[str, dict[str, float]] = {
    "SVM": {"sentiment": 62.9, "hate": 36.7, "offensive": 52.3},
    "FastText": {"sentiment": 62.9, "hate": 50.6, "offensive": 73.4},
    "BLSTM": {"sentiment": 58.3, "hate": 52.6, "offensive": 71.7},
    "RoBERTa-Base (RoB-Bs)": {"sentiment": 71.3, "hate": 46.6, "offensive": 79.5},
    "RoBERTa-Retrained (RoB-RT)": {"sentiment": 72.6, "hate": 52.3, "offensive": 80.5},
    "RoBERTa-Twitter (RoB-Tw)": {"sentiment": 69.1, "hate": 46.9, "offensive": 77.1},
}
README_LEADERBOARD: dict[str, dict[str, float]] = {
    "BERTweet": {"sentiment": 73.4, "hate": 56.4, "offensive": 79.5},
    "TimeLMs-2021": {"sentiment": 73.7, "hate": 55.1, "offensive": 82.2},
}
README_DISAGREES_WITH_PAPER = {
    "RoBERTa-Retrained (RoB-RT) sentiment": {"paper_table3": 72.6, "readme": 72.8},
    "RoBERTa-Twitter (RoB-Tw) hate": {"paper_table3": 46.9, "readme": 49.9},
}
"""The README leaderboard at :data:`UPSTREAM_COMMIT` versus the paper; the paper is used."""

BATCH = 2 * social_pulse.PER_CHANNEL
"""Posts per simulated sweep: one name's X plus Reddit fetch."""
BRIGADE_ACCOUNTS = 3
"""Accounts posting each abusive line in the brigade scenario — the coordination threshold."""
MAX_CLEAN_WITHHELD = 0.06
"""The selection constraint: a screen may withhold at most this share of clean posts."""
BOOTSTRAP = 1000
SEED = 20260925
CI_TARGET = 0.03
"""Half-width of the 95% interval a run of the Qwen analyst should achieve to be worth reading."""
FLOOR = 1 / 3
"""Macro-recall of any trivial classifier on three classes — and of the crowd read before it
scored tone (2026-09-26). Gap closures are measured from here, so they do not shrink merely
because the crowd read itself moved."""

CANDIDATES: tuple[AbuseScreen, ...] = (
    AbuseScreen(obscenity=True, cuss=False, threats=False),
    AbuseScreen(obscenity=False, cuss=True, threats=False),
    AbuseScreen(obscenity=True, cuss=True, threats=False),
    DEFAULT_SCREEN,
)
"""The matcher combinations the selection chooses between: each lexicon alone, both, and both
with the threat patterns (production). Together they are also the ablation."""

WITHOUT_ALLOWLIST = AbuseScreen(allowlist=False)
"""Reported, not a selection candidate. The market allowlist exists for text TweetEval does not
contain — "top gainers & losers", "dumb money" — so a general-Twitter benchmark cannot judge it;
its effect is measured on TweetEval (how many posts it excuses) and on the desk's own snapshot of
real X and Reddit posts (how many false withholdings it prevents). On the first validation pass
it was the variant the criterion picked, by a single offensive validation tweet that happened to
contain an allowlisted market phrase; a criterion that cannot see the market text is the wrong
judge of a rule written for it."""


class TweetEvalDataError(RuntimeError):
    """The data is missing or misaligned. Raised, never scored around."""


@dataclass(frozen=True)
class Split:
    task: str
    split: str
    texts: tuple[str, ...]
    labels: tuple[int, ...]
    names: dict[int, str]

    @property
    def classes(self) -> tuple[int, ...]:
        return tuple(sorted(self.names))

    def class_counts(self) -> dict[str, int]:
        return {self.names[c]: sum(1 for y in self.labels if y == c) for c in self.classes}


def tweeteval_dir(explicit: Path | None = None) -> Path:
    return explicit or Path(os.environ.get("ARGUS_TWEETEVAL_DIR") or DEFAULT_TWEETEVAL)


def _lines(path: Path) -> list[str]:
    """One record per ``\\n`` — not ``str.splitlines``, which would also split a tweet on a
    U+2028 or form feed inside it and silently shift every later label by one."""
    with path.open(encoding="utf-8", newline=None) as handle:
        lines = handle.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def git_blob_sha1(path: Path) -> str:
    """The hash git gives the file, line endings normalised to LF as the upstream tree stores
    them (a Windows checkout rewrites them to CRLF)."""
    body = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def load_split(root: Path, task: str, split: str) -> Split:
    """Texts and labels aligned by line, refusing to proceed if they are not."""
    text_path = root / task / f"{split}_text.txt"
    label_path = root / task / f"{split}_labels.txt"
    mapping_path = root / task / "mapping.txt"
    for path in (text_path, label_path, mapping_path):
        if not path.is_file():
            raise TweetEvalDataError(f"{path} is missing; point ARGUS_TWEETEVAL_DIR at a "
                                     f"checkout of {UPSTREAM}/datasets")
    texts = _lines(text_path)
    raw_labels = _lines(label_path)
    if len(texts) != len(raw_labels):
        raise TweetEvalDataError(f"{task}/{split}: {len(texts)} texts but {len(raw_labels)} "
                                 "labels — refusing to score a misaligned file")
    names: dict[int, str] = {}
    for row in _lines(mapping_path):
        key, _, name = row.partition("\t")
        if key.strip():
            names[int(key)] = name.strip()
    labels = tuple(int(value) for value in raw_labels)
    unknown = sorted(set(labels) - set(names))
    if unknown:
        raise TweetEvalDataError(f"{task}/{split}: labels {unknown} are not in mapping.txt")
    return Split(task=task, split=split, texts=tuple(texts), labels=labels, names=names)


def provenance(root: Path, files: Iterable[str],
               expected: dict[str, str] | None = None) -> dict[str, Any]:
    pinned = UPSTREAM_BLOBS if expected is None else expected
    out: dict[str, Any] = {}
    for name in files:
        sha = git_blob_sha1(root / name)
        out[name] = {"git_blob_sha1": sha, "matches_upstream": sha == pinned.get(name)}
    return out


REFERENCE_BLOBS: dict[str, str] = {
    "predictions/sentiment.txt": "1027371e3a8386cfd176603e4279c201e5d026fa",
    "predictions/hate.txt": "aa5f386c91c8bd1282bced9f2edc24b28d0f1e5d",
    "predictions/offensive.txt": "e1c3330c660ce8d7291d6143031ae0420cf34bcd",
}
"""The repository's own test-set predictions, which its README (line 45) says "correspond to the
best model evaluated in the paper, i.e., RoBERTa re-trained on Twitter (RoB-Rt in the paper)".
They are one label per line and no text, so the strongest cited baseline is *re-scored* here with
this module's metric on the same items as every ARGUS row, rather than only quoted from a table.
One run, where Table 3 reports the mean of three, so a small difference from Table 3 is expected
and is published, not smoothed. Blob hashes from the upstream tree at :data:`UPSTREAM_COMMIT`
(``gh api``, 2026-09-25)."""
REFERENCE_NAME = "RoBERTa-Retrained (RoB-RT), TweetEval's published test predictions"


def reference_predictions(root: Path, split: Split) -> list[int]:
    """RoB-RT's published labels for ``split``, aligned by line with the gold labels."""
    path = root.parent / "predictions" / f"{split.task}.txt"
    if not path.is_file():
        raise TweetEvalDataError(f"{path} is missing; it ships beside {UPSTREAM}/datasets")
    labels = [int(value) for value in _lines(path)]
    if len(labels) != len(split.labels):
        raise TweetEvalDataError(f"{path.name}: {len(labels)} predictions but "
                                 f"{len(split.labels)} gold labels — misaligned")
    unknown = sorted(set(labels) - set(split.names))
    if unknown:
        raise TweetEvalDataError(f"{path.name}: labels {unknown} are not in mapping.txt")
    return labels


# =============================================================================================
# Metrics, reimplemented from the definitions (TweetEval §2.2; SemEval-2017 Task 4 §4.1, eq. 1).
# =============================================================================================


def confusion(gold: Sequence[int], pred: Sequence[int],
              classes: Sequence[int]) -> dict[int, dict[int, int]]:
    """``matrix[true][predicted]``."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold labels but {len(pred)} predictions")
    matrix = {c: dict.fromkeys(classes, 0) for c in classes}
    for truth, guess in zip(gold, pred, strict=True):
        matrix[truth][guess] += 1
    return matrix


def per_class(matrix: dict[int, dict[int, int]]) -> dict[int, dict[str, float]]:
    """Precision, recall and F1 per class; an undefined ratio is 0, as scikit-learn reports it
    by default (the implementation TweetEval's own script calls)."""
    out: dict[int, dict[str, float]] = {}
    for c in matrix:
        tp = matrix[c][c]
        predicted = sum(matrix[t][c] for t in matrix)
        actual = sum(matrix[c].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def macro_recall(gold: Sequence[int], pred: Sequence[int], classes: Sequence[int]) -> float:
    """AvgRec: recall averaged over the classes, each class weighted equally."""
    scores = per_class(confusion(gold, pred, classes))
    return sum(s["recall"] for s in scores.values()) / len(scores)


def macro_f1(gold: Sequence[int], pred: Sequence[int], classes: Sequence[int]) -> float:
    scores = per_class(confusion(gold, pred, classes))
    return sum(s["f1"] for s in scores.values()) / len(scores)


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    return sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / len(gold)


Metric = Callable[[Sequence[int], Sequence[int], Sequence[int]], float]


def bootstrap_ci(gold: Sequence[int], pred: Sequence[int], classes: Sequence[int],
                 metric: Metric, *, resamples: int = BOOTSTRAP,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile 95% interval over test items resampled with replacement."""
    rng = random.Random(seed)
    n = len(gold)
    values = []
    for _ in range(resamples):
        index = rng.choices(range(n), k=n)
        values.append(metric([gold[i] for i in index], [pred[i] for i in index], classes))
    values.sort()
    return values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]


def paired_bootstrap(gold: Sequence[int], first: Sequence[int], second: Sequence[int],
                     classes: Sequence[int], metric: Metric, *, resamples: int = BOOTSTRAP,
                     seed: int = SEED) -> dict[str, float]:
    """The difference ``metric(first) - metric(second)`` on the same resampled items, its 95%
    interval, and the share of resamples in which ``first`` does not come out ahead."""
    rng = random.Random(seed)
    n = len(gold)
    diffs = []
    for _ in range(resamples):
        index = rng.choices(range(n), k=n)
        g = [gold[i] for i in index]
        diffs.append(metric(g, [first[i] for i in index], classes)
                     - metric(g, [second[i] for i in index], classes))
    diffs.sort()
    return {
        "difference": metric(gold, first, classes) - metric(gold, second, classes),
        "ci95_low": diffs[int(0.025 * resamples)],
        "ci95_high": diffs[int(0.975 * resamples) - 1],
        "share_of_resamples_first_not_ahead": sum(1 for d in diffs if d <= 0) / resamples,
    }


def sentiment_scores(split: Split, pred: Sequence[int]) -> dict[str, Any]:
    classes = split.classes
    matrix = confusion(split.labels, pred, classes)
    low, high = bootstrap_ci(split.labels, pred, classes, macro_recall)
    return {
        "macro_recall": macro_recall(split.labels, pred, classes),
        "macro_recall_ci95": [low, high],
        "accuracy": accuracy(split.labels, pred),
        "macro_f1": macro_f1(split.labels, pred, classes),
        "recall_by_class": {split.names[c]: s["recall"] for c, s in per_class(matrix).items()},
        "predicted_counts": {split.names[c]: sum(1 for p in pred if p == c) for c in classes},
        "confusion_true_by_predicted": {
            split.names[t]: {split.names[p]: n for p, n in row.items()}
            for t, row in matrix.items()},
    }


# =============================================================================================
# Scorers.
# =============================================================================================


def _posts(texts: Sequence[str], now: datetime, *, start: int = 0,
           authors: Sequence[str] | None = None,
           minutes_apart: int = 1) -> list[social_pulse.Post]:
    return [social_pulse.Post(id=f"te{start + i}", claim=text,
                              source=authors[i] if authors else f"@acct{start + i}",
                              available_at=now - timedelta(minutes=minutes_apart * i),
                              channel="x")
            for i, text in enumerate(texts)]


@dataclass(frozen=True)
class CrowdRead:
    """What the real ``pulse()`` said about every tweet, screened and unscreened."""

    pred: list[int]
    """The class the pulse published per tweet; a withheld tweet has none and reads neutral."""
    unscreened: list[int]
    """The same tweets through ``pulse(screen=None)`` — VADER alone, if the pulse is faithful."""
    kept: list[int]
    """Indices of the tweets the abuse screen let through."""
    extra: dict[str, Any]


def _shares(labels: Sequence[str]) -> dict[str, float]:
    return {name: labels.count(name) / len(labels) for name in social_pulse.TONES}


def mean_ci(values: Sequence[float], *, resamples: int = BOOTSTRAP,
            seed: int = SEED) -> dict[str, Any]:
    """The mean of per-sweep values and its percentile 95% interval, sweeps resampled."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return {"mean": sum(values) / n, "ci95": [means[int(0.025 * resamples)],
                                              means[int(0.975 * resamples) - 1]], "sweeps": n}


def crowd_read(split: Split, now: datetime, *,
               tone: social_pulse.Tone | None = None) -> CrowdRead:
    """Run the real ``pulse()`` over the tweets a sweep at a time, with the abuse screen and
    without it, and read back the tone it published — per post and per sweep.

    Per sweep, three differences in the share a reader is shown are recorded against the gold
    shares of *every* tweet in the sweep (what the whole crowd said): the screened read, the
    unscreened read, and screened minus unscreened, which isolates the screen. Weighted minus
    unweighted isolates the coordination weighting."""
    by_name = {name: c for c, name in split.names.items()}
    n = len(split.texts)
    pred = [by_name["neutral"]] * n
    unscreened = [by_name["neutral"]] * n
    kept: list[int] = []
    withheld_by_class = dict.fromkeys(split.names.values(), 0)
    withheld_as_vader = dict.fromkeys(social_pulse.TONES, 0)
    sweeps: dict[str, list[float]] = {}
    coordinated = 0
    for start in range(0, n, BATCH):
        batch = split.texts[start:start + BATCH]
        posts = _posts(batch, now, start=start)
        row = social_pulse.pulse("TWEETEVAL", posts, [], now, tone=tone)
        raw = social_pulse.pulse("TWEETEVAL", posts, [], now, screen=None, tone=tone)
        for read in (row, raw):
            if read.get("tone", {}).get("status") != "ok":
                raise TweetEvalDataError(f"the pulse scored no tone: {read.get('tone')}")
        if row["posts"] + row["withheld_abusive"] != len(batch):
            raise TweetEvalDataError("pulse lost posts: counted plus withheld is not the batch")
        scored = {entry["id"]: entry["label"] for entry in row["post_tone"]}
        everything = {entry["id"]: entry["label"] for entry in raw["post_tone"]}
        if len(scored) != row["posts"] or len(everything) != len(batch):
            raise TweetEvalDataError("the pulse did not publish a tone for every post it kept")
        coordinated += row["tone"]["coordinated_stories_counted_once"]
        gold = []
        for offset, post in enumerate(posts):
            index = start + offset
            gold.append(split.names[split.labels[index]])
            unscreened[index] = by_name[everything[post.id]]
            if post.id in scored:
                pred[index] = by_name[scored[post.id]]
                kept.append(index)
            else:
                withheld_by_class[gold[-1]] += 1
                withheld_as_vader[everything[post.id]] += 1
        truth = _shares(gold)
        shown, full = row["tone"]["share"], raw["tone"]["share"]
        for name in ("negative", "positive"):
            if shown is not None:
                sweeps.setdefault(f"{name}_screened_minus_gold", []).append(
                    shown[name] - truth[name])
                sweeps.setdefault(f"{name}_screened_minus_unscreened", []).append(
                    shown[name] - full[name])
                sweeps.setdefault(f"{name}_weighted_minus_unweighted", []).append(
                    shown[name] - row["tone"]["share_unweighted"][name])
            sweeps.setdefault(f"{name}_unscreened_minus_gold", []).append(
                full[name] - truth[name])
    counts = split.class_counts()
    withheld = n - len(kept)
    kept_gold = [split.names[split.labels[i]] for i in kept]
    all_gold = [split.names[y] for y in split.labels]
    return CrowdRead(pred=pred, unscreened=unscreened, kept=kept, extra={
        "withheld_by_the_abuse_screen": withheld,
        "withheld_share": withheld / n,
        "withheld_share_by_sentiment_class": {
            name: withheld_by_class[name] / counts[name] if counts[name] else None
            for name in counts},
        "withheld_scored_as": "neutral — the pulse publishes no tone for a withheld post",
        "tone_bias_from_the_abuse_screen": {
            "gold_share_all_tweets": _shares(all_gold),
            "gold_share_tweets_the_screen_kept": _shares(kept_gold),
            "withheld_tweets_as_vader_would_have_read_them": withheld_as_vader,
            "per_sweep_share_difference": {key: mean_ci(values)
                                           for key, values in sorted(sweeps.items())},
            "coordinated_stories_counted_once": coordinated,
        },
    })


class _BatchPipeline(Protocol):
    def __call__(self, texts: list[str], *, batch_size: int,
                 truncation: bool) -> list[dict[str, Any]]: ...


FINBERT_TO_TWEETEVAL = {"negative": "negative", "neutral": "neutral", "positive": "positive"}


FINBERT_CHUNK = 512
"""Tweets per pipeline call, and per write of the progress file."""


def _load_progress(cache: Path | None, text_blob: str, n: int) -> tuple[list[int], float]:
    """Labels already computed for this exact test file (``-1`` where not yet), and the seconds
    they took. A file for different text, or of the wrong length, is ignored, not trusted."""
    if cache is None or not cache.is_file():
        return [-1] * n, 0.0
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [-1] * n, 0.0
    labels = saved.get("labels") if isinstance(saved, dict) else None
    if saved.get("text_blob") != text_blob or not isinstance(labels, list) or len(labels) != n:
        return [-1] * n, 0.0
    return [int(v) for v in labels], float(saved.get("seconds") or 0.0)


def finbert_predictions(split: Split, *, batch_size: int = 32, check_batching: int = 100,
                        cache: Path | None = None,
                        text_blob: str = "") -> tuple[list[int], dict[str, Any]]:
    """Every test tweet through the real ``ProsusAI/finbert`` pipeline, offline.

    Batched with attention masks and sorted by length so a batch pads little. Batching is not
    assumed to be harmless: the first ``check_batching`` tweets are also run one at a time and
    the label agreement is reported.

    On a CPU shared with other work finBERT ran at 75 ms a tweet on 2026-09-25 — about fifteen
    minutes for the split — so progress is written through to ``cache`` after every chunk
    (predicted class indices and elapsed seconds only; no text), keyed by the test file's git
    blob hash. An interrupted run resumes where it stopped instead of starting over.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from argus.eval.baselines.finbert_loader import load_finbert

    loaded = load_finbert()
    # The loader types its return as a one-argument callable; the object is the transformers
    # text-classification pipeline, whose documented call also takes batch_size and truncation.
    pipe = cast(_BatchPipeline, loaded)
    by_name = {name: c for c, name in split.names.items()}
    n = len(split.texts)
    labels, elapsed = _load_progress(cache, text_blob, n)
    resumed = sum(1 for label in labels if label >= 0)
    order = sorted((i for i in range(n) if labels[i] < 0), key=lambda i: len(split.texts[i]))
    for begin in range(0, len(order), FINBERT_CHUNK):
        chunk = order[begin:begin + FINBERT_CHUNK]
        started = time.perf_counter()
        outputs = pipe([split.texts[i] for i in chunk], batch_size=batch_size, truncation=True)
        for i, output in zip(chunk, outputs, strict=True):
            labels[i] = by_name[FINBERT_TO_TWEETEVAL[str(output["label"])]]
        elapsed += time.perf_counter() - started
        if cache is not None:
            cache.write_text(json.dumps({"text_blob": text_blob, "labels": labels,
                                         "seconds": elapsed}), encoding="utf-8")
    sample = list(split.texts[:check_batching])
    single = [by_name[FINBERT_TO_TWEETEVAL[str(o["label"])]] for o in loaded(sample)]
    model = getattr(loaded, "model", None)
    config = getattr(model, "config", None)
    return labels, {
        "model": "ProsusAI/finbert (local Hugging Face cache, offline)",
        "model_commit": str(getattr(config, "_commit_hash", None) or "unknown"),
        "label_map": FINBERT_TO_TWEETEVAL,
        "seconds": elapsed,
        "ms_per_tweet": elapsed / n * 1000,
        "tweets_resumed_from_an_interrupted_run": resumed,
        "batching_label_agreement": {
            "tweets": len(sample),
            "agree": sum(1 for a, b in zip(single, labels[:len(sample)], strict=True) if a == b),
        },
    }


def vader_predictions(split: Split) -> tuple[list[int], dict[str, Any]]:
    from argus.eval.baselines import vader_loader

    analyzer = vader_loader.load_analyzer()
    by_name = {name: c for c, name in split.names.items()}
    started = time.perf_counter()
    labels = [by_name[vader_loader.label(float(analyzer.polarity_scores(text)["compound"]))]
              for text in split.texts]
    elapsed = time.perf_counter() - started
    return labels, {**vader_loader.provenance(), "seconds": elapsed,
                    "ms_per_tweet": elapsed / len(split.texts) * 1000}


def analyst_not_run(finbert_ci: Sequence[float], n: int) -> dict[str, Any]:
    """What a run of the Qwen analyst would take, from the measured spread — not a guess."""
    half = (finbert_ci[1] - finbert_ci[0]) / 2
    needed = int(n * (half / CI_TARGET) ** 2) + 1 if half > 0 else None
    return {
        "status": "not_run",
        "reason": "agents/analysts.SentimentAnalyst answers through Qwen, and this item's Qwen "
                  "budget is zero calls; no response was requested or recorded.",
        "qwen_calls": 0,
        "how_it_would_be_scored": "each tweet as one Evidence item; bullish -> positive, "
                                  "bearish -> negative, neutral and insufficient_evidence -> "
                                  "neutral",
        "tweets_needed_for_a_95pct_interval_of_plus_minus": CI_TARGET,
        "tweets_needed": needed,
        "derived_from": f"finBERT's measured interval half-width {half:.4f} on {n} tweets, "
                        "scaled by 1/sqrt(n); one Qwen call per tweet",
    }


# =============================================================================================
# The abuse screen: selection on validation, then test, then the pulse itself.
# =============================================================================================


def screen_scores(screen: Callable[[str], Any], split: Split) -> dict[str, Any]:
    """TweetEval's macro-F1, plus the two rates the pulse operates on."""
    pred = [1 if screen(text).abusive else 0 for text in split.texts]
    classes = split.classes
    scores = per_class(confusion(split.labels, pred, classes))
    positives = sum(1 for y in split.labels if y == 1)
    negatives = len(split.labels) - positives
    return {
        "macro_f1": macro_f1(split.labels, pred, classes),
        "abusive_withheld_share": scores[1]["recall"],
        "clean_withheld_share": sum(1 for y, p in zip(split.labels, pred, strict=True)
                                    if y == 0 and p == 1) / negatives,
        "f1_by_class": {split.names[c]: s["f1"] for c, s in scores.items()},
        "abusive_posts": positives,
        "clean_posts": negatives,
    }


def ordinary_withheld(screen: Callable[[str], Any], split: Split) -> float:
    return sum(1 for text in split.texts if screen(text).abusive) / len(split.texts)


def select(val: dict[str, dict[str, Any]]) -> str:
    """The declared criterion: most abusive posts withheld, within both clean-post limits."""
    eligible = {name: row for name, row in val.items()
                if row["offensive"]["clean_withheld_share"] <= MAX_CLEAN_WITHHELD
                and row["sentiment_withheld_share"] <= MAX_CLEAN_WITHHELD}
    if not eligible:
        raise TweetEvalDataError("no candidate screen meets the clean-post limits")
    return max(eligible, key=lambda name: (eligible[name]["mean_abusive_withheld_share"],
                                           -eligible[name]["offensive"]["clean_withheld_share"]))


def screen_table(splits: dict[str, Split], sentiment: Split,
                 screens: Sequence[AbuseScreen] = CANDIDATES) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for candidate in screens:
        hate = screen_scores(candidate, splits["hate"])
        offensive = screen_scores(candidate, splits["offensive"])
        table[candidate.name] = {
            "hate": hate,
            "offensive": offensive,
            "mean_abusive_withheld_share": (hate["abusive_withheld_share"]
                                            + offensive["abusive_withheld_share"]) / 2,
            "sentiment_withheld_share": ordinary_withheld(candidate, sentiment),
        }
    return table


def _pipeline_run(texts: Sequence[str], labels: Sequence[int], *, brigade: bool,
                  screened: bool, now: datetime) -> dict[str, Any]:
    """Feed labelled tweets through ``pulse()`` and ``lines_for()`` a sweep at a time; count
    what abusive text got counted and what got quoted. Counts only — no text leaves here."""
    screen = DEFAULT_SCREEN if screened else None
    if brigade:
        chosen = [(t, y) for t, y in zip(texts, labels, strict=True) if y == 1]
        per_batch = BATCH // BRIGADE_ACCOUNTS
        batches = [chosen[i:i + per_batch] for i in range(0, len(chosen), per_batch)]
    else:
        pairs = list(zip(texts, labels, strict=True))
        batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]
    totals = dict.fromkeys(("posts", "abusive_posts", "abusive_posts_counted",
                            "clean_posts", "clean_posts_withheld", "abusive_texts_in_snapshot",
                            "abusive_texts_quoted_to_reader", "coordinated_stories_counted",
                            "coordinated_abusive_stories_in_snapshot"), 0)
    for number, batch in enumerate(batches):
        if brigade:
            lines = [t for t, _ in batch for _ in range(BRIGADE_ACCOUNTS)]
            tags = [y for _, y in batch for _ in range(BRIGADE_ACCOUNTS)]
            authors = [f"@b{number}_{i}" for i in range(len(lines))]
        else:
            lines = [t for t, _ in batch]
            tags = [y for _, y in batch]
            authors = [f"@o{number}_{i}" for i in range(len(lines))]
        posts = _posts(lines, now, authors=authors, minutes_apart=2 if brigade else 1)
        row = social_pulse.pulse("TWEETEVAL", posts, [], now, screen=screen)
        flags = [bool(screen and screen(text).abusive) for text in lines]
        if row["posts"] != flags.count(False):
            raise TweetEvalDataError("pulse's count disagrees with the screen it was given")
        label_of = {text[:220]: y for text, y in zip(lines, tags, strict=True)}
        snapshot = {"generated_at": now.isoformat(), "coordination_rule": "",
                    "symbols": [row]}
        console = "\n".join(social_pulse.lines_for("TWEETEVAL", snapshot, now=now,
                                                   screen=screen))
        totals["posts"] += len(lines)
        totals["abusive_posts"] += tags.count(1)
        totals["clean_posts"] += tags.count(0)
        totals["abusive_posts_counted"] += sum(1 for y, f in zip(tags, flags, strict=True)
                                               if y == 1 and not f)
        totals["clean_posts_withheld"] += sum(1 for y, f in zip(tags, flags, strict=True)
                                              if y == 0 and f)
        totals["coordinated_stories_counted"] += row["coordinated_stories"]
        for story in row["top_stories"]:
            if label_of.get(story["text"]) != 1:
                continue
            totals["abusive_texts_in_snapshot"] += 1
            if story["text"][:110] in console or story["text"][:140] in console:
                totals["abusive_texts_quoted_to_reader"] += 1
            if story["coordinated"]:
                totals["coordinated_abusive_stories_in_snapshot"] += 1
    return totals


def pipeline_check(split: Split, now: datetime) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario, brigade in (("organic", False), ("brigade", True)):
        out[scenario] = {
            "unscreened_as_before_2026_09_25": _pipeline_run(
                split.texts, split.labels, brigade=brigade, screened=False, now=now),
            "screened": _pipeline_run(split.texts, split.labels, brigade=brigade,
                                      screened=True, now=now),
        }
    return out


def live_snapshot_check(path: Path = social_pulse.PULSE_PATH) -> dict[str, Any]:
    """The desk's own last X and Reddit snapshot: how many of its published story texts the
    screen would now withhold, and how many the market allowlist keeps. Counts only."""
    snapshot = social_pulse.load(path)
    if snapshot is None:
        return {"status": "no snapshot on disk"}
    texts = [str(s.get("text", "")) for row in snapshot.get("symbols", [])
             for s in row.get("top_stories", [])]
    reasons: dict[str, int] = {}
    for text in texts:
        verdict = DEFAULT_SCREEN(text)
        if verdict.abusive and verdict.reason:
            reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
    return {"snapshot_generated_at": snapshot.get("generated_at"), "story_texts": len(texts),
            "would_be_withheld": sum(reasons.values()), "by_reason": reasons,
            "would_be_withheld_without_the_market_allowlist": sum(
                1 for text in texts if WITHOUT_ALLOWLIST(text).abusive)}


def allowlist_effect(splits: Iterable[Split]) -> dict[str, int]:
    """Posts per split that the market allowlist keeps from being withheld."""
    return {f"{split.task}/{split.split}": sum(
        1 for text in split.texts
        if WITHOUT_ALLOWLIST(text).abusive and not DEFAULT_SCREEN(text).abusive)
        for split in splits}


# =============================================================================================
# The artefact: counts and scores only.
# =============================================================================================


def _strings(blob: Any) -> Iterable[str]:
    if isinstance(blob, str):
        yield blob
    elif isinstance(blob, dict):
        for key, value in blob.items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(blob, list):
        for value in blob:
            yield from _strings(value)


def assert_no_text(report: dict[str, Any], texts: Iterable[str], *, shortest: int = 20) -> None:
    """Refuse to write an artefact that carries any benchmark text of ``shortest`` characters or
    more. TweetEval has no licence; its words stay where they are."""
    leaves = [s for s in _strings(report) if len(s) >= shortest]
    for text in texts:
        if len(text) >= shortest and any(text in leaf for leaf in leaves):
            raise TweetEvalDataError("the report contains TweetEval text; not writing it")


def paper_rows(task: str) -> dict[str, float]:
    rows = {name: scores[task] / 100 for name, scores in PAPER_TABLE3_TEST.items()}
    rows.update({f"{name} (README leaderboard)": scores[task] / 100
                 for name, scores in README_LEADERBOARD.items()})
    return rows


def _rank(value: float, rows: dict[str, float]) -> dict[str, Any]:
    below = sorted(name for name, score in rows.items() if score < value)
    return {"beats": below, "beats_count": len(below), "of": len(rows),
            "gap_to_best": max(rows.values()) - value}


def sentiment_verdict(scorers: dict[str, dict[str, Any]], rows: dict[str, float],
                      paired: dict[str, dict[str, float]] | None = None) -> str:
    crowd_row = scorers["argus_crowd_read"]
    crowd = crowd_row["macro_recall"]
    fin = scorers["finbert"]["macro_recall"]
    vad = scorers["vader"]["macro_recall"]
    best_name = max(rows, key=lambda name: rows[name])
    weakest_name = min(rows, key=lambda name: rows[name])
    standing = ("LOSES to every published baseline" if crowd < rows[weakest_name]
                else "does not lose to every published baseline")
    ci = crowd_row.get("macro_recall_ci95") or [crowd, crowd]
    verdict = (f"ARGUS {standing} on sentiment classification. Its crowd read now scores tone "
               f"with VADER on every post the abuse screen keeps: macro-recall {crowd:.3f} "
               f"(95% CI {ci[0]:.3f}-{ci[1]:.3f}), up from the {FLOOR:.3f} floor it scored with "
               f"no polarity. VADER alone scores {vad:.3f}, finBERT (the capability's named "
               f"baseline) {fin:.3f}, the paper's weakest baseline {weakest_name} "
               f"{rows[weakest_name]:.3f} and the best reported model {best_name} "
               f"{rows[best_name]:.3f}.")
    reference = scorers.get("tweeteval_reference")
    if paired:
        against_vader = paired["argus_crowd_read_minus_vader"]
        against_finbert = paired["argus_crowd_read_minus_finbert"]
        verdict += (f" Against VADER alone it is {against_vader['difference']:+.3f} (paired 95% "
                    f"CI {against_vader['ci95_low']:+.3f} to {against_vader['ci95_high']:+.3f}): "
                    f"that is the abuse screen's cost, since a withheld post gets no tone. "
                    f"Against finBERT it is {against_finbert['difference']:+.3f} "
                    f"({against_finbert['ci95_low']:+.3f} to {against_finbert['ci95_high']:+.3f}).")
    if reference:
        verdict += (f" RoB-RT, re-scored here at {reference['macro_recall']:.3f}, is still "
                    f"{reference['macro_recall'] - crowd:.3f} ahead: a tweet-domain transformer "
                    f"closes that at the cost of torch in the runtime.")
    verdict += (" ARGUS's LLM analyst was not run (Qwen budget zero).")
    return verdict


def gap_closers(scorers: dict[str, dict[str, Any]], rows: dict[str, float]) -> list[dict[str, Any]]:
    """What would close the sentiment gap, each option priced by a number measured in this run or
    cited from the paper — never by a number nobody measured.

    Closures are measured from :data:`FLOOR`, where the crowd read stood before it scored tone,
    so the four options stay comparable with the run that first priced them. VADER is the one
    adopted (2026-09-26); its row says so and gives what the crowd read scores with it.
    """
    crowd = FLOOR
    now_scores = scorers["argus_crowd_read"]["macro_recall"]
    vader = scorers["vader"]["macro_recall"]
    finbert = scorers["finbert"]["macro_recall"]
    best_name = max(rows, key=lambda name: rows[name])
    tweet_models = {name: score for name, score in rows.items() if "RoBERTa" in name
                    or "BERTweet" in name or "TimeLMs" in name}
    analyst = scorers["argus_sentiment_analyst"]
    reference = scorers.get("tweeteval_reference")
    return [
        {
            "option": "a tweet-domain fine-tuned transformer (TweetEval's own RoBERTa checkpoints, "
                      "BERTweet or TimeLMs) as the polarity reader",
            "evidence": ("RoB-RT's published test predictions re-scored here; the others cited"
                         if reference else "cited, not run here"),
            "macro_recall": {"lowest_of_them": min(tweet_models.values()),
                             "best_reported": rows[best_name], "best_reported_model": best_name,
                             "rob_rt_rescored_here": reference["macro_recall"]
                             if reference else None},
            "closes_of_the_gap_to_best": 1.0,
            "cost": "puts torch and a 500 MB checkpoint into the desk's runtime, which "
                    "pyproject.toml keeps dependency-light by standing rule; not run here because "
                    "no checkpoint is cached on this machine and nothing may be downloaded",
        },
        {
            "option": "a measured run of ARGUS's own Sentiment Integrity Analyst (Qwen)",
            "evidence": "not run: this item's Qwen budget is zero",
            "macro_recall": None,
            "closes_of_the_gap_to_best": None,
            "cost": f"{analyst['tweets_needed']} Qwen calls for a 95% interval of "
                    f"plus or minus {CI_TARGET}; the literature it cites (BloombergGPT trailing an "
                    "always-neutral predictor, FinMA at chance) is reason not to assume it wins",
        },
        {
            "option": "finBERT, the capability's named baseline, as the polarity reader",
            "evidence": "measured on every test tweet in this run",
            "macro_recall": finbert,
            "closes_of_the_gap_to_best": (finbert - crowd) / (rows[best_name] - crowd),
            "cost": "torch in the runtime, and it is a financial-news model: on tweets it trails "
                    "the lexicon scorer below",
        },
        {
            "option": "VADER (MIT, pure Python, no model) as the polarity reader",
            "status": "adopted in market/social_pulse.py on 2026-09-26, weighted so a "
                      "coordinated story counts once",
            "evidence": "measured on every test tweet in this run",
            "macro_recall": vader,
            "argus_crowd_read_with_it": now_scores,
            "closes_of_the_gap_to_best": (vader - crowd) / (rows[best_name] - crowd),
            "crowd_read_closes_of_the_gap_to_best": (now_scores - crowd)
                                                    / (rows[best_name] - crowd),
            "cost": "no new heavy dependency; still below "
                    + ", ".join(sorted(name for name, score in rows.items() if score > vader))
                    + " — and a polarity score is exactly the reading a coordinated campaign "
                      "inflates, which is why the crowd read counts a coordinated story once",
        },
    ]


def tone_bias_verdict(bias: dict[str, Any]) -> str:
    """Whether the abuse screen's skew toward negative posts moves the tone a reader is shown,
    stated from the per-sweep differences — the sign and interval decide the wording."""
    diff = bias["per_sweep_share_difference"]
    screen = diff["negative_screened_minus_unscreened"]
    vs_gold = diff["negative_screened_minus_gold"]
    raw_vs_gold = diff["negative_unscreened_minus_gold"]
    weighting = diff["negative_weighted_minus_unweighted"]
    low, high = screen["ci95"]
    moved = "lowers" if high < 0 else "raises" if low > 0 else "does not measurably move"
    return (
        f"The abuse screen {moved} the negative share a reader is shown: per 60-post sweep, "
        f"screened minus unscreened is {screen['mean']:+.2%} (95% CI {low:+.2%} to "
        f"{high:+.2%}, {screen['sweeps']} sweeps). Gold negative share is "
        f"{bias['gold_share_all_tweets']['negative']:.1%} of all tweets and "
        f"{bias['gold_share_tweets_the_screen_kept']['negative']:.1%} of the ones kept. Against "
        f"the gold share of the whole sweep, VADER's negative share is "
        f"{raw_vs_gold['mean']:+.2%} unscreened and {vs_gold['mean']:+.2%} screened, so the "
        f"screen adds to a bias VADER already has. The withheld tweets, had they been scored, "
        f"would have read {bias['withheld_tweets_as_vader_would_have_read_them']}. The "
        f"coordination weighting moved the negative share by {weighting['mean']:+.2%} on this "
        f"organic benchmark ({bias['coordinated_stories_counted_once']} coordinated stories "
        f"formed in all sweeps), as expected where nobody is brigading.")


def abuse_verdict(test: dict[str, Any], pipeline: dict[str, Any],
                  reference: dict[str, dict[str, Any]] | None = None) -> str:
    hate = test["hate"]
    offensive = test["offensive"]
    brigade = pipeline["hate"]["brigade"]
    organic_quoted = sum(pipeline[task]["organic"]["screened"]["abusive_texts_quoted_to_reader"]
                         for task in pipeline)
    verdict = (
        f"Before the screen, social_pulse counted every abusive post and quoted "
        f"{brigade['unscreened_as_before_2026_09_25']['abusive_texts_quoted_to_reader']} "
        f"coordinated abusive HatEval lines back to the reader in the brigade scenario; with "
        f"it, {brigade['screened']['abusive_texts_quoted_to_reader']}. The screen withholds "
        f"{hate['abusive_withheld_share']:.1%} of hateful and "
        f"{offensive['abusive_withheld_share']:.1%} of offensive test posts, and "
        f"{offensive['clean_withheld_share']:.1%} of clean ones. Macro-F1 "
        f"{hate['macro_f1']:.3f} on hate and {offensive['macro_f1']:.3f} on offensive: a "
        f"word screen, so hostility with no listed word in it passes and is still counted. "
        f"Brigade quotes fall to zero because the console no longer quotes any coordinated "
        f"story's text (the screen alone left every sweep quoting a missed line); in the "
        f"organic scenario {organic_quoted} missed abusive line(s) were still quoted as the "
        f"most-shared story.")
    if reference:
        verdict += (
            f" TweetEval's RoB-RT, re-scored on the same items, reaches "
            f"{reference['hate']['macro_f1']:.3f} and {reference['offensive']['macro_f1']:.3f}: "
            f"the regression check still finds a defect — abusive posts the screen misses are "
            f"counted as crowd volume — and what would close it is a trained classifier, which "
            f"the runtime's no-torch rule currently excludes.")
    return verdict


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - long local run
    import argparse

    from argus.eval.artefact import write

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tweeteval", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--finbert-cache", type=Path, default=None,
                        help="progress file for finBERT's labels (class indices only, no text)")
    args = parser.parse_args(argv)
    root = tweeteval_dir(args.tweeteval)
    now = datetime(2026, 9, 25, 12, tzinfo=UTC)

    files = [f"{task}/{name}" for task in ("sentiment", "hate", "offensive")
             for name in ("mapping.txt", "test_text.txt", "test_labels.txt", "val_text.txt",
                          "val_labels.txt")]
    data = provenance(root, files)
    sentiment = load_split(root, "sentiment", "test")
    val = {task: load_split(root, task, "val") for task in ("hate", "offensive", "sentiment")}
    test = {task: load_split(root, task, "test") for task in ("hate", "offensive")}
    print(f"data: {sum(v['matches_upstream'] for v in data.values())}/{len(data)} files match "
          f"upstream {UPSTREAM_COMMIT[:7]}")

    # --- the abuse screen: select on validation, then test -----------------------------------
    val_table = screen_table({"hate": val["hate"], "offensive": val["offensive"]},
                             val["sentiment"])
    chosen = select(val_table)
    test_table = screen_table(test, sentiment, (*CANDIDATES, WITHOUT_ALLOWLIST))
    print(f"screen selected on validation: {chosen}; production: {DEFAULT_SCREEN.name}")
    pipeline = {task: pipeline_check(test[task], now) for task in ("hate", "offensive")}
    hate_ci = bootstrap_ci(test["hate"].labels,
                           [1 if DEFAULT_SCREEN(t).abusive else 0 for t in test["hate"].texts],
                           (0, 1), macro_f1)
    off_ci = bootstrap_ci(test["offensive"].labels,
                          [1 if DEFAULT_SCREEN(t).abusive else 0
                           for t in test["offensive"].texts], (0, 1), macro_f1)

    # --- sentiment ---------------------------------------------------------------------------
    read = crowd_read(sentiment, now)
    crowd, crowd_extra = read.pred, read.extra
    vader, vader_extra = vader_predictions(sentiment)
    crowd_extra["unscreened_pulse_agrees_with_vader_alone"] = {
        "tweets": len(vader),
        "agree": sum(1 for a, b in zip(read.unscreened, vader, strict=True) if a == b)}
    kept_split = Split(task="sentiment", split="test, tweets the abuse screen kept",
                       texts=tuple(sentiment.texts[i] for i in read.kept),
                       labels=tuple(sentiment.labels[i] for i in read.kept),
                       names=sentiment.names)
    crowd_on_kept = sentiment_scores(kept_split, [crowd[i] for i in read.kept])
    vader_on_kept = macro_recall(kept_split.labels, [vader[i] for i in read.kept],
                                 kept_split.classes)
    print(f"crowd read {macro_recall(sentiment.labels, crowd, sentiment.classes):.4f}  "
          f"vader {macro_recall(sentiment.labels, vader, sentiment.classes):.4f}")
    finbert, finbert_extra = finbert_predictions(
        sentiment, cache=args.finbert_cache,
        text_blob=git_blob_sha1(root / "sentiment" / "test_text.txt"))
    print(f"finbert {macro_recall(sentiment.labels, finbert, sentiment.classes):.4f}", flush=True)
    majority = [max(sentiment.classes, key=lambda c: sentiment.labels.count(c))] * len(
        sentiment.labels)
    reference = {task: reference_predictions(root, split)
                 for task, split in (("sentiment", sentiment), *test.items())}
    data.update(provenance(root.parent, REFERENCE_BLOBS, REFERENCE_BLOBS))

    scorers = {
        "argus_crowd_read": {
            **sentiment_scores(sentiment, crowd), **crowd_extra,
            "what": "market/social_pulse.pulse — VADER tone per post the abuse screen keeps, "
                    "the class read back from the pulse's own post_tone; a withheld post is "
                    "scored neutral",
            "before_2026_09_26": {"macro_recall": FLOOR, "what": "assigned no polarity"},
            "on_the_tweets_it_scored": {
                "tweets": len(read.kept),
                "macro_recall": crowd_on_kept["macro_recall"],
                "macro_recall_ci95": crowd_on_kept["macro_recall_ci95"],
                "vader_alone_on_the_same_tweets": vader_on_kept,
                "class_counts": kept_split.class_counts()}},
        "finbert": {**sentiment_scores(sentiment, finbert), **finbert_extra},
        "vader": {**sentiment_scores(sentiment, vader), **vader_extra},
        "majority_class": sentiment_scores(sentiment, majority),
        "tweeteval_reference": {
            **sentiment_scores(sentiment, reference["sentiment"]), "what": REFERENCE_NAME,
            "paper_table3": PAPER_TABLE3_TEST["RoBERTa-Retrained (RoB-RT)"]["sentiment"] / 100,
            "readme": README_DISAGREES_WITH_PAPER["RoBERTa-Retrained (RoB-RT) sentiment"][
                "readme"] / 100},
    }
    scorers["argus_sentiment_analyst"] = analyst_not_run(
        scorers["finbert"]["macro_recall_ci95"], len(sentiment.labels))
    rows = paper_rows("sentiment")
    for name in ("argus_crowd_read", "finbert", "vader", "majority_class",
                 "tweeteval_reference"):
        scorers[name]["against_paper_baselines"] = _rank(scorers[name]["macro_recall"], rows)
    print(f"RoB-RT re-scored {scorers['tweeteval_reference']['macro_recall']:.4f}", flush=True)
    paired = {
        "finbert_minus_vader": paired_bootstrap(sentiment.labels, finbert, vader,
                                                sentiment.classes, macro_recall),
        "finbert_minus_argus_crowd_read": paired_bootstrap(sentiment.labels, finbert, crowd,
                                                           sentiment.classes, macro_recall),
        "rob_rt_minus_vader": paired_bootstrap(sentiment.labels, reference["sentiment"], vader,
                                               sentiment.classes, macro_recall),
        "rob_rt_minus_argus_crowd_read": paired_bootstrap(
            sentiment.labels, reference["sentiment"], crowd, sentiment.classes, macro_recall),
        "argus_crowd_read_minus_vader": paired_bootstrap(sentiment.labels, crowd, vader,
                                                         sentiment.classes, macro_recall),
        "argus_crowd_read_minus_finbert": paired_bootstrap(sentiment.labels, crowd, finbert,
                                                           sentiment.classes, macro_recall),
    }
    print(f"crowd read (tone) {scorers['argus_crowd_read']['macro_recall']:.4f} "
          f"{scorers['argus_crowd_read']['macro_recall_ci95']}", flush=True)
    production = {task: [1 if DEFAULT_SCREEN(t).abusive else 0 for t in test[task].texts]
                  for task in ("hate", "offensive")}
    reference_abuse = {
        task: {
            "what": REFERENCE_NAME,
            "macro_f1": macro_f1(test[task].labels, reference[task], (0, 1)),
            "macro_f1_ci95": list(bootstrap_ci(test[task].labels, reference[task], (0, 1),
                                               macro_f1)),
            "paper_table3": PAPER_TABLE3_TEST["RoBERTa-Retrained (RoB-RT)"][task] / 100,
            "abusive_posts_it_flags": sum(1 for y, p in zip(test[task].labels, reference[task],
                                                            strict=True) if y == 1 and p == 1),
            "abusive_posts": test[task].labels.count(1),
            "rob_rt_minus_production_screen": paired_bootstrap(
                test[task].labels, reference[task], production[task], (0, 1), macro_f1),
        }
        for task in ("hate", "offensive")}

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "qwen_calls": 0,
        "data": {
            "source": f"{UPSTREAM} at {UPSTREAM_COMMIT}, read in place from a local clone",
            "licence": "none published upstream; data read in place, never copied, and no "
                       "benchmark text appears in this file",
            "files": data,
        },
        "sentiment": {
            "metric": "macro-averaged recall (AvgRec), TweetEval §2.2 / SemEval-2017 Task 4 §4.1",
            "why_not_accuracy": {
                "majority_class_accuracy": scorers["majority_class"]["accuracy"],
                "majority_class_macro_recall": scorers["majority_class"]["macro_recall"],
                "class_shares": {k: v / len(sentiment.labels)
                                 for k, v in sentiment.class_counts().items()},
            },
            "test_tweets": len(sentiment.labels),
            "class_counts": sentiment.class_counts(),
            "scorers": scorers,
            "paired_bootstrap_macro_recall": paired,
            "paper_baselines_macro_recall": rows,
            "paper": PAPER,
            "readme_disagrees_with_paper": README_DISAGREES_WITH_PAPER,
            "verdict": sentiment_verdict(scorers, rows, paired),
            "what_would_close_the_gap": gap_closers(scorers, rows),
            "tone_bias_from_the_abuse_screen": tone_bias_verdict(
                crowd_extra["tone_bias_from_the_abuse_screen"]),
        },
        "abuse_screen": {
            "production_screen": DEFAULT_SCREEN.name,
            "lexicon": {
                "obscenity_phrases": len(abuse.OBSCENITY),
                "cuss_rated_2_entries": len(abuse.CUSS_RATED_2),
                "cuss_rated_2_sha256": abuse.lexicon_digest(abuse.CUSS_RATED_2),
                "cuss_rated_2_sha256_pinned": abuse.CUSS_RATED_2_SHA256,
                "threat_patterns": len(abuse.THREAT_PATTERNS),
                "market_allowlist_phrases": len(abuse.MARKET_ALLOWLIST),
            },
            "selection": {
                "criterion": "largest mean of hate-class and offensive-class recall (share of "
                             "abusive posts withheld) on the validation splits, subject to "
                             f"withholding at most {MAX_CLEAN_WITHHELD:.0%} of clean OffensEval "
                             f"posts and at most {MAX_CLEAN_WITHHELD:.0%} of ordinary sentiment "
                             "tweets; fixed after a first validation pass",
                "validation": val_table,
                "selected": chosen,
                "selected_is_production": chosen == DEFAULT_SCREEN.name,
            },
            "test": test_table,
            "production_macro_f1_ci95": {"hate": list(hate_ci), "offensive": list(off_ci)},
            "tweeteval_reference_rescored": reference_abuse,
            "hate_negative_class_is_not_clean": (
                "HatEval's negative class is 'not-hate', not 'not-offensive': "
                f"{test_table[DEFAULT_SCREEN.name]['hate']['clean_withheld_share']:.1%} of those "
                "posts carry a listed profanity, slur or threat and are withheld by design. The "
                "clean-post rate is the OffensEval 'not-offensive' row and the ordinary "
                "sentiment tweets, which is where the selection limit is applied."),
            "paper_baselines_macro_f1": {"hate": paper_rows("hate"),
                                         "offensive": paper_rows("offensive")},
            "against_paper_baselines": {
                "hate": _rank(test_table[DEFAULT_SCREEN.name]["hate"]["macro_f1"],
                              paper_rows("hate")),
                "offensive": _rank(test_table[DEFAULT_SCREEN.name]["offensive"]["macro_f1"],
                                   paper_rows("offensive")),
            },
            "pipeline_regression": pipeline,
            "live_snapshot": live_snapshot_check(),
            "market_allowlist_excused": allowlist_effect(
                [sentiment, *test.values(), *val.values()]),
            "verdict": abuse_verdict(test_table[DEFAULT_SCREEN.name], pipeline, reference_abuse),
        },
    }
    assert_no_text(report, [*sentiment.texts, *test["hate"].texts, *test["offensive"].texts,
                            *val["hate"].texts, *val["offensive"].texts,
                            *val["sentiment"].texts])
    undefined = write(args.out, report)
    print(report["sentiment"]["verdict"])
    print(report["abuse_screen"]["verdict"])
    print(f"written to {args.out}" + (f" (null: {undefined})" if undefined else ""))
    return 0


__all__ = [
    "CANDIDATES",
    "PAPER_TABLE3_TEST",
    "README_LEADERBOARD",
    "REFERENCE_BLOBS",
    "REPORT_PATH",
    "WITHOUT_ALLOWLIST",
    "Split",
    "TweetEvalDataError",
    "accuracy",
    "allowlist_effect",
    "assert_no_text",
    "bootstrap_ci",
    "confusion",
    "crowd_read",
    "gap_closers",
    "git_blob_sha1",
    "live_snapshot_check",
    "load_split",
    "macro_f1",
    "macro_recall",
    "main",
    "paired_bootstrap",
    "per_class",
    "pipeline_check",
    "reference_predictions",
    "screen_scores",
    "select",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
