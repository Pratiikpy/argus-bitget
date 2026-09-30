"""ARGUS's trader memory (`lui/memory.py`) vs. real, installed mem0ai 2.2.1 — same inputs, both
systems, both run for real, per `research/harvest/04-mem0.md` §8 ("Same-input head-to-head plan").

**What was taken from where.** The experiment design (fixed profiles P1-P3, the four-session
shape, the three "mem0 should structurally win" cases, the recall/contradiction/latency/cost
metric split) is §8's own plan, followed as written. P4 and every fixture's exact wording is new,
written here after §8 was read and before either side ran (the honest tuning/held-out split §21
requires — this file has no regex-tuning step to protect, since `lui/memory.py` is never edited by
this module or by this task, but the split is kept anyway so a held-out generalisation number
exists rather than being asserted). `TUNING_STATEMENTS`, `NOT_ABOUT_ME` and `TUNING_TASKS` are
`eval/memory_eval.py`'s own 30/20/20 fixtures, imported unchanged (`memory_eval.py:34-116`) —
"extended rather than replaced" per §8. Every mem0 mechanism cited below is read from the
INSTALLED package (`~/.venvs/mem0/Lib/site-packages/mem0`, mem0ai 2.2.1, pip),
not the GitHub source `04-mem0.md` read — see `eval/baselines/mem0_runner.py`'s own docstring for
the file:line cross-check against that note (its `_add_to_vector_store` line numbers matched
within single digits; the "V3 PHASED BATCH PIPELINE" ADD-only architecture and the exactly-one
`generate_response()` call per `add()` were independently reconfirmed by reading and grepping the
installed copy here, not assumed from the note).

**What was measured, per §8's own three numbered items (the fourth and fifth — latency and cost —
folded together below as the task instructions group them):**

1. **Recall and precision of stated personal facts**, tuning and held-out separately, hand-labelled
   ground truth (every `Statement.mem0_expected` entry below is hand-written against the exact
   text it grades, not derived), plus the `NOT_ABOUT_ME` false-positive set (also tuning +
   held-out), plus the three "structurally mem0 wins" cases §8 names, each independently
   RE-VERIFIED against real, installed `argus.lui.memory.extract()` before being written into a
   fixture here (see :data:`TUNING_STRUCTURAL`/:data:`HELD_OUT_STRUCTURAL` docstrings for the
   exact verification run — this caught that §8's own suggested "two theses in one message"
   example is no longer a gap: `extract()`'s `_THESIS` loop already uses `.finditer()`, not
   `.search()`, since 2026-09-28 per its own docstring; the single-value kinds —
   budget/max_loss/horizon/style/capital — still use `.search()` and still only catch the first of
   two same-kind facts in one message, which is what the structural fixtures here actually probe
   instead).
2. **Contradiction handling** — two independent checks, both against real code, no reimplementation
   of either system's logic: (a) a deterministic, network-free check of
   `argus.lui.memory.merge()`'s own "newer fact of the same kind replaces the older, records what
   it replaced" behaviour against real mem0 `add()`+`get_all()` output (mem0's `main.py` never
   deletes a superseded memory — confirmed here by listing the store after both a contradicted
   style and the fact it replaced were added, not assumed from `research/harvest/04-mem0.md` §5's
   own citation of this); (b) a live, real `argus.lui.server.handle_ask()` round trip (state a
   mandate, ask an IMPACT question, contradict the mandate, ask again) confirming the mandate
   check's own verdict flips between the two answers and the `Remembered:` line names what it
   replaced — real code, live Bitget candle data, run for profiles P3 and P4. mem0's side of (b) —
   "does the answering LLM state the current fact without blending both" — was NOT run: it would
   need a second LLM call per contradiction case to synthesise an answer from mem0's retrieved
   memories, which mem0 does not do itself (its own `search()` returns ranked memories, not a
   synthesised answer) and building that synthesis step is out of this module's scope. **NOT
   VERIFIED**: whether an LLM answering from mem0's retrieved memories would blend the old and new
   fact; what IS verified is whether mem0's own ranking (`search()`, `_search_vector_store`) puts
   the newer memory ahead of the superseded one, and whether both remain retrievable via
   `get_all()`.
3. **Latency per turn and cost (LLM calls/tokens) per turn**, both sides, measured, not estimated:
   ARGUS's `extract()`/`merge()`/`handle_ask()` wall-clock is timed directly; mem0's is timed inside
   `mem0_runner.py`, around the real `Memory.add()`/`search()`/`get_all()` calls only (subprocess
   spawn and `Memory()` construction overhead are reported separately, since a live long-running
   service would pay that once, not per turn — see :func:`latency_and_cost`'s docstring). Token
   counts are mem0's own real `response.usage` from the Qwen endpoint's OpenAI-compatible response
   (via `response_callback`, `mem0/configs/llms/openai.py:35`), not estimated from the ~500-line
   prompt's approximate length the way `research/harvest/04-mem0.md` §8 item 5 had to.

**Qwen budget.** Capped at 400 calls (`mem0_runner.MAX_QWEN_CALLS`, enforced in the subprocess
too, not just here, so a malformed request fails loudly rather than overspending); this run's
`main()` computes and prints `planned_adds` before spending anything, and the artefact's own
`qwen_calls_used`/`qwen_calls_planned` fields carry the actual count. `search()`/`get_all()` cost
zero Qwen calls — confirmed by reading `mem0/memory/main.py`: neither method's implementation
calls `self.llm` anywhere (grepped the full body of both).

**What this module does NOT claim.** Metric §8-1, "profile-consistency of the verdict"
(`eval/profilestudy.py`'s methodology, `desk/personalisation.judge()` across profiles on identical
market state), is explicitly out of the three-item scope this task named and is not scored here as
a head-to-head — mem0 has no verdict-producing mechanism to run it against at all (§8 says so
itself). The live mandate round trip built for contradiction-metric (b) above happens to also
demonstrate a verdict flip (a real "no" becomes a real "yes" on the same TSLA question once the
mandate contradicts itself) — that is reported as supplementary colour under the contradiction
metric, clearly labelled ARGUS-only, not folded into a comparison mem0 cannot have.

This module does not write, touch or import anything under `eval/capabilities/` or
`eval/standing.py` — no OWNED/TIED/LOST verdict is asserted here; that call is left to whoever
reads this artefact.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.eval import memory_eval
from argus.eval.baselines import mem0_loader
from argus.lui.memory import extract, merge
from argus.truth.artefact import write
from argus.truth.paths import DATA_DIR

REPORT_PATH = DATA_DIR / "memory_comparison.json"

MAX_WORKERS = 6
"""Chains are sharded across this many threads in the mem0 subprocess; see
`eval/baselines/mem0_runner.py`'s docstring for why each worker owns its own `Memory` instance."""


# =================================================================================================
# Fixtures — frozen before either side ran. TUNING reuses `eval/memory_eval.py` unchanged (imported,
# not copied); HELD_OUT is new, written after TUNING was decided, never fed back into any tuning
# step (there is none here — `lui/memory.py` is not edited by this task — kept anyway for the
# discipline: a number from data neither side has seen is worth more than one from data it has).
# =================================================================================================


@dataclass(frozen=True)
class Statement:
    """One message, hand-labelled with what a correct extraction of it contains.

    ``argus_kind``: the single `Fact.kind` ARGUS's `extract()` is expected to produce, or ``None``
    for a sentence that states nothing about the trader (the `NOT_ABOUT_ME` sets) or for a
    structural case where ARGUS is expected to extract nothing at all.

    ``mem0_expected``: one entry per DISTINCT fact mem0 should capture from this exact text, each
    entry a tuple of substrings where ANY one appearing (case-insensitive, anywhere in the
    concatenation of every memory string mem0's `add()` returned for this statement) counts that
    fact as recalled; ALL entries must be satisfied for the statement to count as fully recalled.
    Hand-written per statement against its own text — not derived from ARGUS's kind label or from
    any shared heuristic, so a statement ARGUS fails to extract at all still carries a real,
    independently-written expectation for mem0's side.
    """

    text: str
    argus_kind: str | None
    mem0_expected: tuple[tuple[str, ...], ...]


TUNING_STATEMENTS: tuple[Statement, ...] = tuple(
    Statement(text=text, argus_kind=kind, mem0_expected=(kw,))
    for (text, kind), kw in zip(memory_eval.STATEMENTS, (
        ("10",), ("5",), ("15",), ("8",), ("20",),
        ("20",), ("30",),
        ("swing",), ("day",), ("long",), ("week",), ("month",),
        ("conservative",), ("aggressive",), ("risk-averse", "risk averse"),
        ("earnings",), ("momentum",), ("crypto",),
        ("50",), ("250",), ("20",),
        ("NVDA",), ("BTC",), ("TSLA",), ("gold", "XAU"), ("ETH",),
        ("TSLA",), ("DOGE",), ("MSTR",), ("COIN",),
    ), strict=True)
)
"""`eval/memory_eval.py`'s own 30 `(text, kind)` pairs (`memory_eval.py:34-65`), unchanged, each
paired here with a hand-written mem0 keyword expectation for the same statement."""


TUNING_STRUCTURAL: tuple[Statement, ...] = (
    # Verified against real, installed `argus.lui.memory.extract()` before being written here:
    #   extract("What is my risk tolerance? I think I can handle it but my max loss should "
    #           "probably be around 10%") == []
    # The leading question word ("What") trips `_ASKING` (`memory.py:105-108`), and because the
    # message does not begin with "I" the WHOLE extraction is abandoned — the real max_loss
    # statement later in the same message is never reached. `.search()` never runs at all here,
    # so this is a stronger case than a `.search()`-vs-`.finditer()` gap: nothing downstream of the
    # guard gets a chance.
    Statement(
        text="What is my risk tolerance? I think I can handle it but my max loss should "
             "probably be around 10%",
        argus_kind=None, mem0_expected=(("10",),),
    ),
    # Verified: extract("My risk budget is 20% -- actually you know what, no single name above "
    #                    "30% of my risk") == [Fact(kind="budget", value="0.2", ...)] — ONE fact.
    # `_BUDGET.search()` (`memory.py:72-74`) returns its first match and stops; the message states
    # TWO budget-shaped facts (a flat 20% and a 30%-per-name cap) and only the first is kept.
    Statement(
        text="My risk budget is 20% -- actually you know what, no single name above 30% of "
             "my risk",
        argus_kind="budget", mem0_expected=(("20",), ("30",)),
    ),
    # Verified: extract("I am saving up for a house down payment next year so I want to be "
    #                    "careful with this account") == [] — no KINDS regex targets a savings goal.
    Statement(
        text="I am saving up for a house down payment next year so I want to be careful "
             "with this account",
        argus_kind=None, mem0_expected=(("house", "down payment", "home"),),
    ),
)
"""The three cases §8 names as ones "mem0 should structurally win" — an unstructured fact with no
regex kind, two facts of the same kind in one message, and a fact `_ASKING`'s leading-question
guard zeroes out entirely. All three independently re-verified against real, installed `extract()`
before being written above (§8's own suggested multi-thesis example was checked too and found
already fixed — see this module's own docstring)."""


NOT_ABOUT_ME: tuple[str, ...] = memory_eval.NOT_ABOUT_ME
"""`eval/memory_eval.py`'s own 20 sentences (`memory_eval.py:68-89`), unchanged."""


HELD_OUT_STATEMENTS: tuple[Statement, ...] = (
    # max_loss — note the first: verified against real extract() to be a genuine MISS (ARGUS's
    # regex requires the verb "lose"; "can't exceed" never reaches it). Kept as a real, unforced
    # held-out result rather than dropped for looking bad — see this module's own report of it.
    Statement("my max drawdown can't exceed 12%", "max_loss", (("12",),)),
    Statement("I won't lose more than 7% on any single trade", "max_loss", (("7",),)),
    Statement("my risk budget is 35%", "budget", (("35",),)),
    Statement("no single position above 25% of my risk", "budget", (("25",),)),
    Statement("Im a position trader", "horizon", (("position",),)),
    Statement("i am a long term investor", "horizon", (("long",),)),
    Statement("I hold for a few hours usually", "horizon", (("hour",),)),
    Statement("I hold for years", "horizon", (("year",),)),
    Statement("im quite cautious", "style", (("cautious",),)),
    Statement("I only trade news", "style", (("news",),)),
    Statement("event-driven swing trader here", "style", (("event", "event-driven"),)),
    Statement("my book is 300k", "capital", (("300",),)),
    Statement("I have around $75k to invest", "capital", (("75",),)),
    Statement("an $80k account is what I run", "capital", (("80",),)),
    Statement("i believe gold will rally hard", "thesis", (("gold", "XAU"),)),
    Statement("I think AAPL will underperform this quarter", "thesis", (("AAPL",),)),
    Statement("I expect SOL to double by year end", "thesis", (("SOL",),)),
    Statement("i dont want any DOGE in my book", "avoid", (("DOGE",),)),
    Statement("I never touch QQQ leveraged products", "avoid", (("QQQ",),)),
    Statement("i wont hold ETH right now", "avoid", (("ETH",),)),
)
"""20 new statements, 2-3 per KIND, new vocabulary vs. `TUNING_STATEMENTS` — written after
`TUNING_STATEMENTS`/`TUNING_STRUCTURAL` above, never fed back into anything. Every one of the 20
was run through real, installed `extract()` before being written here (kind label and value match
what `extract()` actually returned) EXCEPT the first, whose ground truth is a genuine, verified
extraction failure, kept rather than dropped."""


HELD_OUT_STRUCTURAL: tuple[Statement, ...] = (
    # Verified: extract("How do I size this position? I usually hold for weeks and I cant lose "
    #                    "more than 12% total") == [] — same `_ASKING` leading-question guard as
    # the tuning structural case, new wording, different kinds involved (horizon + max_loss).
    Statement(
        text="How do I size this position? I usually hold for weeks and I cant lose more "
             "than 12% total",
        argus_kind=None, mem0_expected=(("week", "weeks"), ("12",)),
    ),
    # Verified: extract("I hold for weeks normally -- well actually for this one Ill hold for "
    #                    "months") == [Fact(kind="horizon", value="168", ...)] — ONE fact (weeks);
    # `_HORIZON.search()` never reaches the second "months" clause.
    Statement(
        text="I hold for weeks normally -- well actually for this one Ill hold for months",
        argus_kind="horizon", mem0_expected=(("week", "weeks"), ("month", "months")),
    ),
    # Verified: extract("Im trying to retire early so every trade needs to pull its weight") == []
    Statement(
        text="Im trying to retire early so every trade needs to pull its weight",
        argus_kind=None, mem0_expected=(("retire", "retirement"),),
    ),
)
"""The held-out mirror of `TUNING_STRUCTURAL`: same three gap shapes, new wording and new kinds
(horizon instead of budget/max_loss for the "two facts" and "leading question" cases), each
independently re-verified against real `extract()` before being written above."""


HELD_OUT_NOT_ABOUT_ME: tuple[str, ...] = (
    "is 35% too much risk for one name",
    "what horizon do position traders usually use",
    "how cautious should a beginner be",
    "is a $300k book considered large",
    "will gold rally this year",
    "why is AAPL underperforming",
    "can SOL double from here",
    "is DOGE worth avoiding",
    "do long term investors use stop losses",
    "what is a good max drawdown limit",
    "I wonder if the fed will cut",
    "should I avoid ETH right now",
    "how much is 75k in BTC terms",
    "is QQQ a good etf",
    "what makes a trader event-driven",
)
"""15 new sentences sharing `HELD_OUT_STATEMENTS`' words but stating nothing about the trader,
same discipline as `NOT_ABOUT_ME`. Each verified against real `extract()` to return `[]` before
being written here."""


TUNING_TASKS: tuple[tuple[str, str, str], ...] = memory_eval.TASKS
"""`eval/memory_eval.py`'s own 20 `(statement, question, marker)` triples (`memory_eval.py:92-116`),
unchanged."""


HELD_OUT_TASKS: tuple[tuple[str, str, str], ...] = (
    ("my max drawdown is 12%", "what if the nasdaq drops 15%? I hold 70% AAPL 30% MSFT",
     "Remembered:"),
    ("my loss limit is 6%", "what if BTC drops 20%? I hold 40% BTC 60% ETH", "Remembered:"),
    ("I hold for a few hours usually", "will TSLA be higher?", "Remembered:"),
    ("i am a long term investor", "odds gold is up", "Remembered:"),
    ("I expect SOL to double by year end", "is SOL overbought", "Your thesis on SOL"),
    ("i believe gold will rally hard", "gold price", "Your thesis on XAU"),
    ("i dont want any DOGE in my book", "DOGE price", "Remembered:"),
    ("no single position above 25% of my risk",
     "I hold 40% NVDA, 60% MSFT — what does adding 25% SOL do to my risk?", "Remembered:"),
)
"""8 new two-session tasks, new statements/questions vs. `TUNING_TASKS`, same shape."""

_HELD_OUT_TASK_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("12",), ("6",), ("hour",), ("long",), ("SOL",), ("gold", "XAU"), ("DOGE",), ("25",),
)
"""Hand-written per `HELD_OUT_TASKS` entry, same convention as `Statement.mem0_expected`."""


# =================================================================================================
# held_out_2 — a SECOND, genuinely blind round (2026-09-30, later the same day).
#
# `HELD_OUT_STATEMENTS`/`HELD_OUT_STRUCTURAL` above stopped being held-out the moment they informed
# a real edit to `lui/memory.py` (sentence-level question handling, in-message corrections recorded
# as `replaces`, a new `goal` kind, wider max_loss/horizon phrasing) — they are TUNING data now,
# whatever their variable names say; `main_held_out_2()` relabels their artefact block
# `held_out_1` to say so explicitly. Everything below was written WITHOUT opening `lui/memory.py`
# or its regexes at all (checked: no `Read`/`Grep`/`Bash cat` touched that file between the fix
# report and this block being saved) — general trading-domain knowledge only, the same way a human
# grader would write ground truth before ever seeing the implementation. `argus_kind="goal"` on the
# life-goal statements is the one exception to "written blind": that literal kind name was stated
# directly by the person who made the fix, not learned by reading source, so using it here is using
# given information, not peeking.
# =================================================================================================

HELD_OUT_2_FROZEN_AT = "2026-09-30T04:40:54.204675+00:00"
"""Captured the moment this block was finished, before `extract()` or mem0 `add()` touched a
single one of its statements — see :func:`freeze_held_out_2`, which writes this timestamp and
every text below into the artefact BEFORE either side runs, so there is a durable record that
nothing here could have been cherry-picked after seeing a result."""

HELD_OUT_2_STATEMENTS: tuple[Statement, ...] = (
    Statement("honestly my max loss cant go past 9% on any one trade", "max_loss", (("9",),)),
    Statement("yo my account's basically 60k rn, nothing fancy", "capital", (("60",),)),
    Statement("quick q -- what counts as overtrading? anyway I usually hold for a couple "
             "months", "horizon", (("month", "months"),)),
    Statement("whats a safe drawdown to aim for? mine is like 7% max i guess", "max_loss",
              (("7",),)),
    Statement("my risk budget's 15% - actually scratch that, 22% feels more right", "budget",
              (("15",), ("22",))),
    Statement("putting money away for my kid's college fund so I keep this account pretty "
             "defensive", "goal", (("college",),)),
    Statement("no more than 4% per name is my rule, learned that the hard way", "budget",
              (("4",),)),
    Statement("I hold stuff for years usually, im not trying to day trade this", "horizon",
              (("year", "years"),)),
    Statement("how do people even size crypto positions? I never go past 10% loss on "
             "BTC-heavy trades", "max_loss", (("10",),)),
    Statement("saving up a down payment for a house so this account has to stay careful",
              "goal", (("house", "down payment"),)),
    Statement("im pretty conservative ngl, dont like big swings", "style", (("conservative",),)),
    Statement("my loss limit's 12%, well actually make that 8% now that I think about it",
              "max_loss", (("12",), ("8",))),
    Statement("trying to build a cushion before I quit my job, so no crazy risks here", "goal",
              (("quit", "job", "cushion"),)),
    Statement("portfolio's around 425k these days", "capital", (("425",),)),
    Statement("whats considered aggressive these days? mine's def aggressive, I go big or go "
             "home", "style", (("aggressive",),)),
    Statement("I trade mostly breakouts, sometimes momentum too", "style",
             (("breakout", "breakouts"),)),
    Statement("cant afford more than 6% down on this one, its a big position", "max_loss",
              (("6",),)),
    Statement("i believe SOL is going to underperform for the rest of the year", "thesis",
              (("SOL",),)),
    Statement("pretty sure XRP rips from here", "thesis", (("XRP",),)),
    Statement("i stay away from AVAX, got burned once", "avoid", (("AVAX",),)),
    Statement("no BNB for me, dont trust it", "avoid", (("BNB",),)),
    Statement("whats a good horizon for swing trades? I hold mine for like 2 weeks typically",
              "horizon", (("week", "weeks"),)),
    Statement("keeping most of this in cash until I retire early, thats the whole point of "
             "the account", "goal", (("retire", "retirement"),)),
    Statement("i think im more of a position trader honestly, holding for a month or two at "
             "a time", "horizon", (("month", "position"),)),
    Statement("my book's maybe 90k right now, started smaller and added to it", "capital",
              (("90",),)),
)
"""25 statements a real trader would type: leading questions followed by a fact (2, 4, 9, 15, 22),
in-message corrections/two-facts-of-one-kind (5, 12), unstructured life-goal facts hypothesised to
land under the new `goal` kind (6, 10, 13, 23), informal spelling throughout ("cant", "im", "rn",
"ngl", no apostrophes), and multi-sentence messages (3, 9, 15, 22). Every kind ARGUS already had is
covered at least twice; `goal` four times."""

HELD_OUT_2_NOT_ABOUT_ME: tuple[str, ...] = (
    "should i be more conservative or does that even matter long term",
    "if my max loss was 20% would that be crazy",
    "my friend says he never holds over the weekend, seems risky to me",
    "some traders swear by a 2% rule per trade",
    "would a 300k account be considered mid-size",
    "what would you do if BTC crashed 30% overnight",
    "isnt a 10% stop pretty tight for crypto",
    "my buddy thinks SOL is going to rip, not sure I agree",
    "how many people actually stick to a max drawdown rule",
    "is retiring early even realistic on a trader's income",
    "what percent of a portfolio should go to one name normally",
    "she said she avoids meme coins entirely, smart move honestly",
    "do most day traders hold positions for hours or minutes",
    "if someone had 50k would you tell them to go aggressive",
    "whats the difference between a swing trader and a position trader anyway",
)
"""15 sentences stating nothing about the trader: questions, hypotheticals, someone else's stated
view (a friend, a buddy, "she said") — the last category deliberately new versus every earlier
NOT_ABOUT_ME set, since a third party's fact is a plausible confusion a sentence-level fix could
introduce if it stopped checking whose statement a sentence reports."""

HELD_OUT_2_TASKS: tuple[tuple[str, str, str], ...] = (
    ("my max loss cant go past 9% on any one trade",
     "what if the nasdaq drops 12%? I hold 80% MSFT 20% AAPL", "Remembered:"),
    ("no more than 4% per name is my rule, learned that the hard way",
     "I hold 70% NVDA 30% AMD -- what does adding 20% SOL do to my risk?", "Remembered:"),
    ("I hold stuff for years usually, im not trying to day trade this", "will gold be higher?",
     "Remembered:"),
    ("im pretty conservative ngl, dont like big swings", "odds TSLA is up", "Remembered:"),
    ("i believe SOL is going to underperform for the rest of the year", "is SOL overbought",
     "Your thesis on SOL"),
    ("pretty sure XRP rips from here", "XRP price", "Your thesis on XRP"),
    ("i stay away from AVAX, got burned once", "AVAX price", "Remembered:"),
    ("whats a good horizon for swing trades? I hold mine for like 2 weeks typically",
     "will ETH go up?", "Remembered:"),
)
"""8 new two-session tasks. The last one states its fact behind a leading question — the exact
shape the "sentence-level question handling" fix targets — so the effect metric measures that fix
end to end (does memory get recorded AND does it change a later answer), not just extraction."""

_HELD_OUT_2_TASK_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("9",), ("4",), ("year", "years"), ("conservative",), ("SOL",), ("XRP",), ("AVAX",),
    ("week", "weeks"),
)
"""Hand-written per `HELD_OUT_2_TASKS` entry, same convention as `Statement.mem0_expected`."""


@dataclass(frozen=True)
class BlindStructuralCase:
    """One of `held_out_2`'s 12 structural cases (4 per gap shape). Unlike `TUNING_STRUCTURAL`'s
    `Statement`, this carries no `argus_kind` pass/fail expectation — the whole point of this round
    is that ARGUS's actual behaviour on these shapes was unknown when they were written (the fix
    that targets them was made but not read), so scoring reports what `extract()` ACTUALLY
    returned (kind, value, text, `replaces`, in full) rather than a boolean tied to an assumption
    about a mechanism never inspected."""

    text: str
    gap: str
    mem0_expected: tuple[tuple[str, ...], ...]


HELD_OUT_2_STRUCTURAL: tuple[BlindStructuralCase, ...] = (
    # --- leading_question_zeroes_extraction (4): a real question, then a real stated fact ---
    BlindStructuralCase(
        "What's a smart position size here? I never risk more than 5% on a single name.",
        "leading_question_zeroes_extraction", (("5",),),
    ),
    BlindStructuralCase(
        "How long should I hold something like this? I usually keep positions for about 3 "
        "months.",
        "leading_question_zeroes_extraction", (("month", "months", "3"),),
    ),
    BlindStructuralCase(
        "Should I diversify more? My portfolio's around 180k right now.",
        "leading_question_zeroes_extraction", (("180",),),
    ),
    BlindStructuralCase(
        "Is momentum trading still worth it? I mostly trade earnings reactions these days.",
        "leading_question_zeroes_extraction", (("earnings",),),
    ),
    # --- two_facts_same_kind_one_message (4): an in-message self-correction ---
    BlindStructuralCase(
        "My loss limit is 10% -- actually, let's make that 6%, feels safer.",
        "two_facts_same_kind_one_message", (("10",), ("6",)),
    ),
    BlindStructuralCase(
        "I usually hold for weeks -- no wait, more like months for this kind of setup.",
        "two_facts_same_kind_one_message", (("week", "weeks"), ("month", "months")),
    ),
    BlindStructuralCase(
        "My risk budget is 25% per name -- hmm, actually 15% is probably wiser.",
        "two_facts_same_kind_one_message", (("25",), ("15",)),
    ),
    BlindStructuralCase(
        "I'm aggressive normally -- but honestly for this account I'd say I'm more "
        "conservative.",
        "two_facts_same_kind_one_message", (("aggressive",), ("conservative",)),
    ),
    # --- unstructured_fact_no_kind (4): a life goal, hypothesised to land under the new `goal` ---
    BlindStructuralCase(
        "Trying to save enough for a down payment on a house, so I'm keeping this account "
        "pretty tame.",
        "unstructured_fact_no_kind", (("house", "down payment"),),
    ),
    BlindStructuralCase(
        "This is basically my retirement cushion, so I don't want to gamble with it.",
        "unstructured_fact_no_kind", (("retirement", "retire"),),
    ),
    BlindStructuralCase(
        "I'm building toward quitting my 9-to-5 in a couple years, so every trade has to "
        "count.",
        "unstructured_fact_no_kind", (("quit", "9-to-5", "job"),),
    ),
    BlindStructuralCase(
        "Putting this aside for my daughter's college fund, nothing risky please.",
        "unstructured_fact_no_kind", (("college",),),
    ),
)
"""4 cases per gap shape, all new wording versus `TUNING_STRUCTURAL`/`HELD_OUT_STRUCTURAL`. No
`argus_kind` ground truth is asserted (see `BlindStructuralCase`'s own docstring) — `mem0_expected`
is still hand-written per case, since mem0's side of this round is an ordinary recall measurement,
not a test of an unread fix."""


# =================================================================================================
# held_out_3 — a THIRD, genuinely blind round (2026-09-30, later the same day again).
#
# `held_out_2` showed the sentence-level/goal/in-message-correction fixes did not generalise past
# the exact `held_out_1` phrasing that prompted them (ARGUS recall fell to 2/25). The coordinator's
# response: `lui/memory_model.py`'s `combined(text, client, now)` — the pattern reader plus, for a
# self-disclosure message, ONE Qwen extraction call whose every fact must quote the message, with
# rejection code for hypotheticals and other people's statements. Everything below was written
# WITHOUT opening `lui/memory_model.py` or `lui/memory.py` — only `lui/server.py`, `lui/router.py`,
# `llm/provider.py` and `llm/qwen.py` were read, to learn the exact call contract
# (`memory_model.combined(text, client, now, price_of=...)`, `client = _model_for(visitor,
# count=False)`, `client.calls`/`client.budget.spent` for cost accounting) — none of those files
# describe what the extractor actually does with a given sentence, so ground truth here is exactly
# as blind as `held_out_2`'s. A throwaway pilot sentence (never a fixture) confirmed the shape
# before any fixture was written: a non-self-disclosure sentence cost 0 calls/0 tokens (a cheap
# pre-gate), a real self-disclosure sentence cost exactly 1 call, ~410 tokens, and its `Fact.text`
# was a verbatim quote from the message — consistent with the coordinator's description, and cited
# here as a fact about the pilot's own two throwaway sentences, not as a claim about this round's
# fixtures, which had not been written yet.
# =================================================================================================

HELD_OUT_3_FROZEN_AT = "2026-09-30T05:10:08.968710+00:00"
"""Captured the moment this block was finished, before `memory_model.combined()` or mem0 `add()`
touched a single one of its statements — see :func:`freeze_held_out_3`."""

HELD_OUT_3_STATEMENTS: tuple[Statement, ...] = (
    Statement("keeping my downside capped at 8%, learned that lesson already", "max_loss",
             (("8",),)),
    Statement("yeah so basically anything past a 6 percent hit and I'm out", "max_loss",
             (("6",),)),
    Statement("risk budget wise I try to stay under 18% total", "budget", (("18",),)),
    Statement("nothing over 12% of the book in one name, thats my rule", "budget", (("12",),)),
    Statement("im in and out same day mostly, dont hold overnight", "horizon", (("day", "days"),)),
    Statement("tend to sit in stuff for like half a year sometimes", "horizon",
             (("half a year", "6 month", "six month", "months"),)),
    Statement("few days tops is my usual hold", "horizon", (("day", "days"),)),
    Statement("pretty risk-on tbh, i chase the big moves", "style",
             (("aggressive", "risk-on", "risk on"),)),
    Statement("im the type that waits for confirmation, not a gambler", "style",
             (("conservative", "cautious", "confirmation"),)),
    Statement("mostly playing earnings drift these days", "style", (("earnings",),)),
    Statement("got about 220k sitting in this account", "capital", (("220",),)),
    Statement("started with like 15k, its grown since", "capital", (("15",),)),
    Statement("think ADA breaks out hard this quarter", "thesis", (("ADA",),)),
    Statement("pretty confident LINK underperforms from here", "thesis", (("LINK",),)),
    Statement("not touching FTT again, obviously", "avoid", (("FTT",),)),
    Statement("staying clear of LUNA, still salty about it", "avoid", (("LUNA",),)),
    Statement("putting a chunk aside every month for my wedding next year", "goal",
             (("wedding",),)),
    Statement("this ones earmarked for an emergency fund, gotta stay boring here", "goal",
             (("emergency fund",),)),
    Statement("trying to hit financial independence by 40, so every position matters", "goal",
             (("financial independence", "FI", "40"),)),
    Statement("honestly just trying to pay off my student loans faster with this", "goal",
             (("student loan", "student loans"),)),
)
"""20 statements, none reused from `TUNING_STATEMENTS`/`HELD_OUT_STATEMENTS`/
`HELD_OUT_2_STATEMENTS`, natural and informal throughout ("tbh", "im", "thats", "ones", no
apostrophes, spelled-out "percent"). `argus_kind` ground truth is hand-written from the text
alone, same convention as every earlier round — the new hybrid extractor's actual recall on each
was unknown when this was written."""

HELD_OUT_3_NOT_ABOUT_ME: tuple[str, ...] = (
    "oh sure, because 10x leverage always works out great",
    "my analyst friend thinks ADA is due for a breakout, whatever that means",
    "some youtuber said LINK is going to zero, classic",
    "what would happen if I went all in on one name",
    "is an 8% stop considered tight or loose these days",
    "he mentioned he never holds through earnings, interesting approach",
    "would 220k be enough to live off dividends",
    "supposedly experts recommend a 6 month emergency fund, is that real",
    "if someone chased every big move, wouldn't they get wrecked eventually",
    "she's convinced FTT is coming back somehow, cant see it myself",
    "how many day traders actually make it past a year",
    "yeah because timing the bottom always works, right",
)
"""12 sentences stating nothing about the trader: sarcasm (1, 12), a friend's/analyst's/an
expert's/a YouTuber's stated view attributed to them by name or relation (2, 3, 6, 8, 10),
hypotheticals (4, 9), and plain questions (5, 7, 11) — exactly the shape the coordinator's own
description says the rejection code targets ("hypotheticals and other people's statements"), so
this is the sharpest test of that specific claim in the whole round."""

HELD_OUT_3_TASKS: tuple[tuple[str, str, str], ...] = (
    ("keeping my downside capped at 8%, learned that lesson already",
     "what if the nasdaq drops 15%? I hold 60% AAPL 40% MSFT", "Remembered:"),
    ("few days tops is my usual hold", "will BTC be higher?", "Remembered:"),
    ("pretty risk-on tbh, i chase the big moves", "odds NVDA is up", "Remembered:"),
    ("think ADA breaks out hard this quarter", "is ADA overbought", "Your thesis on ADA"),
    ("pretty confident LINK underperforms from here", "LINK price", "Your thesis on LINK"),
    ("not touching FTT again, obviously", "FTT price", "Remembered:"),
)
"""6 new two-session tasks, new statements/questions vs. every earlier round."""

_HELD_OUT_3_TASK_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("8",), ("day", "days"), ("aggressive", "risk-on", "risk on", "chase"), ("ADA",), ("LINK",),
    ("FTT",),
)
"""Hand-written per `HELD_OUT_3_TASKS` entry, same convention as `Statement.mem0_expected`."""

HELD_OUT_3_STRUCTURAL: tuple[BlindStructuralCase, ...] = (
    # --- leading_question_zeroes_extraction (3) ---
    BlindStructuralCase(
        "Is a 6-month emergency fund overkill? Either way I'm keeping 25% of this in cash.",
        "leading_question_zeroes_extraction", (("25", "cash"),),
    ),
    BlindStructuralCase(
        "What's too aggressive for a starter account? Mine's about 10k and I'm still learning.",
        "leading_question_zeroes_extraction", (("10",),),
    ),
    BlindStructuralCase(
        "Should I be worried about slippage on small caps? I mostly stick to large caps anyway.",
        "leading_question_zeroes_extraction", (("large cap", "large-cap", "large caps"),),
    ),
    # --- two_facts_same_kind_one_message (3): an in-message self-correction ---
    BlindStructuralCase(
        "My cap is like 20k -- no wait, closer to 30k after that last deposit.",
        "two_facts_same_kind_one_message", (("20",), ("30",)),
    ),
    BlindStructuralCase(
        "I hold for days normally -- actually scratch that, this one's more of a weeks play.",
        "two_facts_same_kind_one_message", (("day", "days"), ("week", "weeks")),
    ),
    BlindStructuralCase(
        "Budget-wise I said 10% earlier but let's bump it to 16%, feeling more confident.",
        "two_facts_same_kind_one_message", (("10",), ("16",)),
    ),
    # --- unstructured_fact_no_kind (3): a life goal ---
    BlindStructuralCase(
        "Every dollar in here is going toward a down payment eventually, so I keep it tame.",
        "unstructured_fact_no_kind", (("down payment",),),
    ),
    BlindStructuralCase(
        "This account's basically my safety net if I ever get laid off.",
        "unstructured_fact_no_kind", (("safety net", "laid off", "layoff"),),
    ),
    BlindStructuralCase(
        "Long game here is just not having to work forever, so patience matters.",
        "unstructured_fact_no_kind",
        (("work forever", "retire", "financial independence", "not working"),),
    ),
)
"""3 cases per gap shape (9 total), all new wording versus every earlier round's structural
fixtures. Same no-`argus_kind`-ground-truth convention as `HELD_OUT_2_STRUCTURAL` (see
`BlindStructuralCase`'s own docstring)."""


# =================================================================================================
# held_out_4 — a FOURTH, genuinely blind round (2026-09-30, later the same day again).
#
# `held_out_3` showed ARGUS's precision was real (0/12 false positives) but recall was thin (5/20)
# — the coordinator's own diagnosis: "the gate and validation were the problem (most misses never
# reached the model)". `lui/memory_model.py` was changed again, using `held_out_3`'s rows, which is
# why `held_out_3` is relabelled tuning below (see `main_held_out_4`). Written the same way as
# `held_out_3`: no `lui/memory_model.py` or `lui/memory.py` read this round either — the call
# contract (`memory_model.combined(text, client, now, price_of=...)`, `_model_for`,
# `client.calls`/`client.budget.spent`) was already learned from `server.py`/`router.py`/
# `provider.py`/`qwen.py` last round and does not need re-reading.
# =================================================================================================

HELD_OUT_4_FROZEN_AT = "2026-09-30T05:30:24.131086+00:00"
"""Captured the moment this block was finished, before `memory_model.combined()` or mem0 `add()`
touched a single one of its statements — see :func:`freeze_held_out_4`."""

HELD_OUT_4_STATEMENTS: tuple[Statement, ...] = (
    # --- no "I" (10): dropped-subject, texting-register self-disclosure ---
    Statement("keeping it under 7% per trade, no exceptions", "max_loss", (("7",),)),
    Statement("nothing above 9% per position, that's the rule here", "budget", (("9",),)),
    Statement("hold for like a week usually, sometimes two", "horizon", (("week", "weeks"),)),
    Statement("in and out within the day mostly", "horizon", (("day", "days"),)),
    Statement("running about 50k in this account", "capital", (("50",),)),
    Statement("convinced DOT breaks higher from here", "thesis", (("DOT",),)),
    Statement("avoiding XRP completely after last time", "avoid", (("XRP",),)),
    Statement("saving toward a car this year, nothing wild please", "goal", (("car",),)),
    Statement("leaning aggressive this cycle tbh", "style", (("aggressive",),)),
    Statement("done with UST, never again", "avoid", (("UST",),)),
    # --- with "I" (10), fresh informal phrasing vs. every earlier round ---
    Statement("I try to keep losses under 11% tops", "max_loss", (("11",),)),
    Statement("I dont go past 16% risk budget usually", "budget", (("16",),)),
    Statement("I sit in trades for months typically", "horizon", (("month", "months"),)),
    Statement("I'm fairly cautious most of the time", "style", (("cautious",),)),
    Statement("I trade a lot of breakout setups", "style", (("breakout", "breakouts"),)),
    Statement("I've got close to 175k in here", "capital", (("175",),)),
    Statement("I think XLM stalls out around here", "thesis", (("XLM",),)),
    Statement("I'm trying to get to a million before 45", "goal", (("million",),)),
    Statement("I keep extra going toward paying off the mortgage faster", "goal",
             (("mortgage",),)),
    Statement("I've got maybe 40k, started small", "capital", (("40",),)),
)
"""20 statements, none reused from any earlier round. The first 10 deliberately drop the subject
pronoun entirely (texting register: "keeping it under 7%", "running about 50k") per the
coordinator's explicit ask for statements with no "I"; the last 10 use "I" with fresh informal
phrasing. `argus_kind` ground truth is hand-written from the text alone — ARGUS's actual recall on
each was unknown when this was written."""

HELD_OUT_4_NOT_ABOUT_ME: tuple[str, ...] = (
    "sure, because averaging down always works out",
    "her advisor keeps pushing more crypto exposure, seems aggressive to me",
    "some influencer swears DOT is the next big thing",
    "what if everyone just held for a decade",
    "is 9% too tight for a stop these days",
    "he apparently never touches leverage, kind of admire that",
    "would 175k be considered a serious account",
    "supposedly a million by 45 is realistic if you save hard, is that true",
    "if someone avoided every red day they'd never trade at all right",
    "he's sure UST bounces back somehow, cant see it",
    "how many people actually stick to their stop losses",
    "oh yeah, because timing every top and bottom is so easy",
)
"""12 sentences stating nothing about the trader: sarcasm (1, 12), an advisor's/an influencer's/a
third person's ("he", "her advisor") stated view (2, 3, 6, 10), hypotheticals (4, 9), plain
questions (5, 7, 8, 11) — same discipline as `HELD_OUT_3_NOT_ABOUT_ME`, new wording throughout."""

HELD_OUT_4_TASKS: tuple[tuple[str, str, str], ...] = (
    ("keeping it under 7% per trade, no exceptions",
     "what if the nasdaq drops 12%? I hold 55% AAPL 45% MSFT", "Remembered:"),
    ("hold for like a week usually, sometimes two", "will SOL be higher?", "Remembered:"),
    ("leaning aggressive this cycle tbh", "odds ETH is up", "Remembered:"),
    ("convinced DOT breaks higher from here", "is DOT overbought", "Your thesis on DOT"),
    ("I think XLM stalls out around here", "XLM price", "Your thesis on XLM"),
    ("avoiding XRP completely after last time", "XRP price", "Remembered:"),
)
"""6 new two-session tasks, including a no-"I" statement (1, 2, 3, 7 — this round's central ask)."""

_HELD_OUT_4_TASK_KEYWORDS: tuple[tuple[str, ...], ...] = (
    ("7",), ("week", "weeks"), ("aggressive",), ("DOT",), ("XLM",), ("XRP",),
)
"""Hand-written per `HELD_OUT_4_TASKS` entry, same convention as `Statement.mem0_expected`."""

HELD_OUT_4_STRUCTURAL: tuple[BlindStructuralCase, ...] = (
    # --- leading_question_zeroes_extraction (3) ---
    BlindStructuralCase(
        "What's the point of diversifying anyway? Either way keeping 30% in stablecoins.",
        "leading_question_zeroes_extraction", (("30", "stablecoin", "stablecoins"),),
    ),
    BlindStructuralCase(
        "Is a starter account even worth it? This one's only 8k for now.",
        "leading_question_zeroes_extraction", (("8",),),
    ),
    BlindStructuralCase(
        "Should slippage even matter at this size? Sticking to large caps mostly.",
        "leading_question_zeroes_extraction", (("large cap", "large-cap", "large caps"),),
    ),
    # --- two_facts_same_kind_one_message (3): an in-message self-correction ---
    BlindStructuralCase(
        "Cap's like 25k -- actually no, closer to 35k with the last add.",
        "two_facts_same_kind_one_message", (("25",), ("35",)),
    ),
    BlindStructuralCase(
        "Hold for a week normally -- wait, this one's more of a month thing.",
        "two_facts_same_kind_one_message", (("week", "weeks"), ("month", "months")),
    ),
    BlindStructuralCase(
        "Said 12% budget earlier -- bumping that to 18%, feeling good about it.",
        "two_facts_same_kind_one_message", (("12",), ("18",)),
    ),
    # --- unstructured_fact_no_kind (3): a life goal ---
    BlindStructuralCase(
        "Every bit extra here goes toward the kids' college eventually.",
        "unstructured_fact_no_kind", (("college",),),
    ),
    BlindStructuralCase(
        "This is the fund for whenever the layoff finally happens.",
        "unstructured_fact_no_kind", (("layoff", "laid off"),),
    ),
    BlindStructuralCase(
        "End goal is just never touching a 9-to-5 again.",
        "unstructured_fact_no_kind", (("9-to-5", "job", "work"),),
    ),
)
"""3 cases per gap shape (9 total), all new wording. Every one of these 9 also drops the subject
pronoun ("Cap's like 25k", "Hold for a week normally"), doubling as more no-"I" coverage."""


@dataclass(frozen=True)
class Profile:
    """A trader's stated mandate, in the two states §8's contradiction test needs: the facts they
    stated first, and the facts that later contradict them, stated in one message each (a natural
    multi-fact turn, not four separate messages) so both sides extract from identical input."""

    name: str
    session1_text: str
    session3_text: str
    mandate_question: str = "should I add 15% TSLA?"


P1 = Profile(
    name="P1_conservative_swing",
    session1_text="I'm conservative. My max loss is 3%. I hold for weeks. My account is $50k.",
    session3_text="",  # P1 never contradicts itself; it is P3's session-1 half.
)
P2 = Profile(
    name="P2_aggressive_day",
    session1_text="I'm aggressive. My max loss is 15%. I hold for days. My account is $150k.",
    session3_text="",
)
P3 = Profile(
    name="P3_contradiction",
    session1_text=P1.session1_text,
    session3_text=P2.session1_text,
)
P4 = Profile(
    name="P4_held_out_contradiction",
    session1_text="I'm risk-averse. My max drawdown is 9%. I usually hold for months. "
                  "My portfolio is $300,000.",
    session3_text="I'm aggressive. My max drawdown is 20%. I hold for a few hours. "
                  "My portfolio is $900,000.",
)
"""P1/P2 verified individually against real `extract()` (all 4 facts each, correct kind and value).
P3 restates P1's exact session-1 text, then P2's exact session-1 text as its own session-3
contradiction — per §8: "states P1's facts in session 1, then P2's facts in session 3". P4 is the
held-out profile, built after P1-P3 above, its own two messages independently verified against real
`extract()` (4/4 facts each) before being written here."""


# =================================================================================================
# ARGUS side — real `argus.lui.memory` functions, no reimplementation, no network.
# =================================================================================================


@dataclass(frozen=True)
class RecallScore:
    total: int
    recalled: int
    missed: tuple[str, ...]
    elapsed_s: float

    @property
    def recall_rate(self) -> float:
        return self.recalled / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "recalled": self.recalled, "recall_rate":
                round(self.recall_rate, 4), "missed": list(self.missed),
                "elapsed_s": round(self.elapsed_s, 6)}


def argus_recall(statements: tuple[Statement, ...]) -> RecallScore:
    """Runs real `extract()` on every statement whose `argus_kind` is set; a statement whose
    `argus_kind` is `None` is a structural case where ARGUS extracting nothing IS the finding, not
    scored as a miss here (it is reported separately, see :func:`structural_cases`)."""
    started = time.perf_counter()
    missed: list[str] = []
    recalled = 0
    total = 0
    for stmt in statements:
        if stmt.argus_kind is None:
            continue
        total += 1
        kinds = [f.kind for f in extract(stmt.text)]
        if stmt.argus_kind in kinds:
            recalled += 1
        else:
            missed.append(stmt.text)
    return RecallScore(total=total, recalled=recalled, missed=tuple(missed),
                       elapsed_s=time.perf_counter() - started)


@dataclass(frozen=True)
class FalsePositiveScore:
    total: int
    false_positives: int
    examples: tuple[tuple[str, tuple[str, ...]], ...]
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {"total": self.total, "false_positives": self.false_positives,
                "false_positive_rate": round(self.false_positives / self.total, 4)
                if self.total else 0.0,
                "examples": [[t, list(k)] for t, k in self.examples],
                "elapsed_s": round(self.elapsed_s, 6)}


def argus_false_positives(sentences: tuple[str, ...]) -> FalsePositiveScore:
    started = time.perf_counter()
    hits: list[tuple[str, tuple[str, ...]]] = []
    for text in sentences:
        kinds = tuple(f.kind for f in extract(text))
        if kinds:
            hits.append((text, kinds))
    return FalsePositiveScore(total=len(sentences), false_positives=len(hits),
                              examples=tuple(hits), elapsed_s=time.perf_counter() - started)


@dataclass(frozen=True)
class StructuralCase:
    text: str
    gap: str
    argus_kinds_found: tuple[str, ...]
    argus_facts_expected: int
    """How many distinct facts a correct extraction of this message contains (from
    `Statement.mem0_expected`'s length) — ARGUS's own count is `len(argus_kinds_found)`, which for
    two of the three gap shapes is fewer."""
    mem0_facts_recalled: int
    mem0_facts_expected: int
    mem0_memories: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "gap": self.gap,
            "argus_kinds_found": list(self.argus_kinds_found),
            "argus_facts_expected": self.argus_facts_expected,
            "argus_facts_found": len(self.argus_kinds_found),
            "mem0_facts_recalled": self.mem0_facts_recalled,
            "mem0_facts_expected": self.mem0_facts_expected,
            "mem0_memories": list(self.mem0_memories),
        }


_GAP_NAMES = ("leading_question_zeroes_extraction", "two_facts_same_kind_one_message",
              "unstructured_fact_no_kind")


def _structural_gap_name(stmt: Statement, index: int) -> str:
    return _GAP_NAMES[index % len(_GAP_NAMES)]


# =================================================================================================
# mem0 side — real, installed mem0ai, run out-of-process. See `mem0_runner.py`/`mem0_loader.py`.
# =================================================================================================


def _matches(memory_texts: str, keywords: tuple[str, ...]) -> bool:
    lowered = memory_texts.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _chain_for_statement(chain_id: str, user_id: str, text: str) -> dict[str, Any]:
    return {"chain_id": chain_id, "user_id": user_id,
           "ops": [{"op": "add", "text": text}]}


def _score_statement_chains(
    statements: tuple[Statement, ...], results_by_id: dict[str, dict[str, Any]], prefix: str,
) -> tuple[RecallScore, list[dict[str, Any]]]:
    recalled = 0
    total = 0
    missed: list[str] = []
    per_statement: list[dict[str, Any]] = []
    for i, stmt in enumerate(statements):
        chain_id = f"{prefix}-{i:04d}"
        chain = results_by_id[chain_id]
        add_result = chain["results"][0] if chain.get("ok") and chain["results"] else None
        memories = [m["memory"] for m in (add_result or {}).get("memories", []) if m.get("memory")]
        combined = " | ".join(memories)
        facts_recalled = sum(1 for group in stmt.mem0_expected if _matches(combined, group))
        full = facts_recalled == len(stmt.mem0_expected) and len(stmt.mem0_expected) > 0
        total += 1
        if full:
            recalled += 1
        else:
            missed.append(stmt.text)
        per_statement.append({
            "text": stmt.text, "argus_kind": stmt.argus_kind,
            "mem0_memories": memories, "facts_expected": len(stmt.mem0_expected),
            "facts_recalled": facts_recalled, "fully_recalled": full,
            "elapsed_s": round((add_result or {}).get("elapsed_s", 0.0), 6),
            "qwen_tokens": ((add_result or {}).get("usage") or {}).get("total_tokens"),
            "op_error": None if (chain.get("ok") and add_result and add_result.get("ok"))
            else (chain.get("error") or (add_result or {}).get("error")),
        })
    score = RecallScore(total=total, recalled=recalled, missed=tuple(missed), elapsed_s=0.0)
    return score, per_statement


def _score_false_positive_chains(
    sentences: tuple[str, ...], results_by_id: dict[str, dict[str, Any]], prefix: str,
) -> tuple[FalsePositiveScore, list[dict[str, Any]]]:
    hits: list[tuple[str, tuple[str, ...]]] = []
    per_sentence: list[dict[str, Any]] = []
    for i, text in enumerate(sentences):
        chain_id = f"{prefix}-{i:04d}"
        chain = results_by_id[chain_id]
        add_result = chain["results"][0] if chain.get("ok") and chain["results"] else None
        memories = [m["memory"] for m in (add_result or {}).get("memories", []) if m.get("memory")]
        if memories:
            hits.append((text, tuple(memories)))
        per_sentence.append({
            "text": text, "mem0_memories": memories, "false_positive": bool(memories),
            "elapsed_s": round((add_result or {}).get("elapsed_s", 0.0), 6),
            "qwen_tokens": ((add_result or {}).get("usage") or {}).get("total_tokens"),
        })
    score = FalsePositiveScore(total=len(sentences), false_positives=len(hits),
                              examples=tuple(hits), elapsed_s=0.0)
    return score, per_sentence


def build_statement_chains(statements: tuple[Statement, ...], prefix: str) -> list[dict[str, Any]]:
    return [_chain_for_statement(f"{prefix}-{i:04d}", f"{prefix}-user-{i:04d}", s.text)
            for i, s in enumerate(statements)]


def build_sentence_chains(sentences: tuple[str, ...], prefix: str) -> list[dict[str, Any]]:
    return [_chain_for_statement(f"{prefix}-{i:04d}", f"{prefix}-user-{i:04d}", text)
            for i, text in enumerate(sentences)]


def build_task_chains(
    tasks: tuple[tuple[str, str, str], ...], prefix: str,
) -> list[dict[str, Any]]:
    """One chain per task: add the statement, search the question (same user); a second chain,
    same index, searches the SAME question under a brand-new, never-populated user — mem0's real
    equivalent of `memory_eval.effect()`'s "ablation" (empty memory) AND "other trader"
    (cross-tenant leak) checks in one, since mem0's architecture has no state between "off" and
    "someone else" (see this module's own docstring for why those two ARGUS-side concepts
    collapse to one here)."""
    chains: list[dict[str, Any]] = []
    for i, (statement, question, _marker) in enumerate(tasks):
        user = f"{prefix}-user-{i:04d}"
        chains.append({
            "chain_id": f"{prefix}-{i:04d}",
            "user_id": user,
            "ops": [{"op": "add", "text": statement},
                    {"op": "search", "query": question, "top_k": 5}],
        })
        chains.append({
            "chain_id": f"{prefix}-fresh-{i:04d}",
            "user_id": f"{prefix}-fresh-user-{i:04d}",
            "ops": [{"op": "search", "query": question, "top_k": 5}],
        })
    return chains


def _score_tasks(
    tasks: tuple[tuple[str, str, str], ...], keywords: tuple[tuple[str, ...], ...], prefix: str,
    results_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """mem0's side of an effect task: does the top hit(s) for the follow-up question carry the
    stated fact's keyword, and does a brand-new user ever carry it (mem0's combined
    ablation/cross-tenant check — see `build_task_chains`'s own docstring for why)."""
    rows = []
    changed = 0
    leaked_or_ablation_carried = 0
    for i, ((statement, question, marker), kw) in enumerate(zip(tasks, keywords, strict=True)):
        main_chain = results_by_id[f"{prefix}-{i:04d}"]
        fresh_chain = results_by_id[f"{prefix}-fresh-{i:04d}"]
        search_op = next((r for r in main_chain.get("results", []) if r.get("op") == "search"),
                         None) if main_chain.get("ok") else None
        fresh_op = (fresh_chain["results"][0]
                   if fresh_chain.get("ok") and fresh_chain["results"] else None)
        hits = search_op.get("results", []) if search_op else []
        top_text = " | ".join(str(h.get("memory", "")) for h in hits[:3])
        carries = _matches(top_text, kw)
        fresh_hits = (fresh_op or {}).get("results", [])
        fresh_carries = bool(fresh_hits)
        if carries:
            changed += 1
        if fresh_carries:
            leaked_or_ablation_carried += 1
        rows.append({"statement": statement, "question": question, "marker": marker,
                    "with_memory_carries": carries, "fresh_user_carries": fresh_carries,
                    "top_hits": hits[:3]})
    return {"tasks": len(tasks), "changed_by_memory": changed,
           "fresh_user_never_carries": leaked_or_ablation_carried == 0,
           "fresh_user_carried_count": leaked_or_ablation_carried, "rows": rows}


def build_profile_chains(profiles: tuple[Profile, ...]) -> list[dict[str, Any]]:
    chains = []
    for p in profiles:
        user = f"profile-{p.name}"
        ops: list[dict[str, Any]] = [{"op": "add", "text": p.session1_text}]
        if p.session3_text:
            ops.append({"op": "add", "text": p.session3_text})
        ops.append({"op": "get_all"})
        ops.append({"op": "search", "query": "what is my risk style and loss limit", "top_k": 10})
        chains.append({"chain_id": f"profile-{p.name}", "user_id": user, "ops": ops})
    return chains


# =================================================================================================
# Contradiction — ARGUS: real merge(); mem0: real add()+get_all()+search().
# =================================================================================================


@dataclass(frozen=True)
class ArgusContradictionResult:
    profile: str
    kinds_checked: tuple[str, ...]
    latest_wins: tuple[bool, ...]
    """One bool per kind: does `get(merged, kind)` equal the SESSION-3 value, not session-1's."""
    replaces_recorded: tuple[bool, ...]
    """One bool per kind: does the surviving `Fact.replaces` name the superseded text."""

    @property
    def all_latest_win(self) -> bool:
        return all(self.latest_wins)

    @property
    def all_replaces_recorded(self) -> bool:
        return all(self.replaces_recorded)

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "kinds_checked": list(self.kinds_checked),
                "latest_wins": list(self.latest_wins), "all_latest_win": self.all_latest_win,
                "replaces_recorded": list(self.replaces_recorded),
                "all_replaces_recorded": self.all_replaces_recorded}


def argus_contradiction(profile: Profile) -> ArgusContradictionResult:
    """`extract()` + `merge()`, exactly as `lui/server.handle_ask` calls them — no network."""
    session1 = extract(profile.session1_text)
    session3 = extract(profile.session3_text)
    merged = merge(session1, session3)
    kinds = tuple(sorted({f.kind for f in session3}))
    latest_wins = []
    replaces_recorded = []
    for kind in kinds:
        old_fact = next((f for f in session1 if f.kind == kind), None)
        new_fact = next((f for f in session3 if f.kind == kind), None)
        surviving = next((f for f in merged if f.kind == kind), None)
        latest_wins.append(
            surviving is not None and new_fact is not None
            and surviving.value == new_fact.value and surviving.text == new_fact.text
        )
        replaces_recorded.append(
            surviving is not None and old_fact is not None
            and old_fact.text[:120] in surviving.replaces
        )
    return ArgusContradictionResult(profile=profile.name, kinds_checked=kinds,
                                    latest_wins=tuple(latest_wins),
                                    replaces_recorded=tuple(replaces_recorded))


@dataclass(frozen=True)
class ArgusMandateFlip:
    """The live `handle_ask()` round trip: state a mandate, ask an IMPACT question, contradict the
    mandate, ask the identical question again — does the verdict actually flip, and does the
    `Remembered:` line say what it replaced. Real network call to live Bitget candles; costs zero
    Qwen tokens (`classified_by` for this question is the static-embedding semantic router,
    `lui/semantic.py`, not an LLM call — confirmed by running this exact round trip with every
    `*QWEN*` environment variable removed beforehand and it still answering in full)."""

    profile: str
    question: str
    before_verdict_line: str
    after_verdict_line: str
    verdict_flipped: bool
    replacement_line_present: bool

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "question": self.question,
                "before_verdict_line": self.before_verdict_line,
                "after_verdict_line": self.after_verdict_line,
                "verdict_flipped": self.verdict_flipped,
                "replacement_line_present": self.replacement_line_present}


def argus_mandate_flip(profile: Profile) -> ArgusMandateFlip:
    from argus.lui.server import handle_ask

    r1 = handle_ask(profile.session1_text, [], visitor=f"memcmp-{profile.name}", memory="")
    r2 = handle_ask(profile.mandate_question, [], visitor=f"memcmp-{profile.name}",
                    memory=r1["memory"])
    before = str(r2.get("lines", [""])[0])
    r3 = handle_ask(profile.session3_text, [], visitor=f"memcmp-{profile.name}",
                    memory=r2["memory"])
    r4 = handle_ask(profile.mandate_question, [], visitor=f"memcmp-{profile.name}",
                    memory=r3["memory"])
    after = str(r4.get("lines", [""])[0])
    yes_no = re.compile(r"\byes\b|\bno\b", re.I)
    before_match = yes_no.search(before)
    after_match = yes_no.search(after)
    before_verdict = before_match.group().lower() if before_match else ""
    after_verdict = after_match.group().lower() if after_match else ""
    flipped = bool(before_verdict) and bool(after_verdict) and before_verdict != after_verdict
    replaced_note = any("replacing" in str(line).lower() for line in r4.get("lines", []))
    return ArgusMandateFlip(profile=profile.name, question=profile.mandate_question,
                            before_verdict_line=before, after_verdict_line=after,
                            verdict_flipped=flipped, replacement_line_present=replaced_note)


@dataclass(frozen=True)
class Mem0ContradictionResult:
    profile: str
    stored_after_both_adds: tuple[str, ...]
    """Every memory `get_all()` returned after BOTH the session-1 and session-3 add() calls."""
    old_fact_still_present: bool
    """Does the OLDER statement's content still appear somewhere in `get_all()` — mem0's own
    "never delete" model, checked empirically rather than assumed from its docs."""
    new_fact_present: bool
    top_search_result: str
    top_result_is_new_not_blended: bool
    """The top-ranked `search()` result names the NEW style word and not the OLD one in the same
    string — "without blending" checked at the level mem0 itself operates on (which single stored
    memory ranks first), not at the level of an LLM-synthesised answer (see this module's own
    docstring for what was NOT run)."""

    def as_dict(self) -> dict[str, Any]:
        return {"profile": self.profile,
                "stored_after_both_adds": list(self.stored_after_both_adds),
                "old_fact_still_present": self.old_fact_still_present,
                "new_fact_present": self.new_fact_present,
                "top_search_result": self.top_search_result,
                "top_result_is_new_not_blended": self.top_result_is_new_not_blended}


_STYLE_WORDS = {"P1_conservative_swing": ("conservative", "aggressive"),
               "P2_aggressive_day": ("aggressive", "conservative"),
               "P3_contradiction": ("conservative", "aggressive"),
               "P4_held_out_contradiction": ("risk-averse", "aggressive")}


def score_mem0_profile(profile: Profile, chain: dict[str, Any]) -> Mem0ContradictionResult:
    results = chain.get("results", []) if chain.get("ok") else []
    # ops order from build_profile_chains: [add(session1), (add(session3) if any,) get_all, search]
    get_all_result = next((r for r in results if r.get("op") == "get_all"), None)
    search_result = next((r for r in results if r.get("op") == "search"), None)
    stored = tuple(m.get("memory", "") for m in (get_all_result or {}).get("results", []))
    combined = " | ".join(stored)
    old_word, new_word = _STYLE_WORDS.get(profile.name, ("", ""))
    old_present = bool(old_word) and old_word.lower() in combined.lower()
    new_present = bool(new_word) and new_word.lower() in combined.lower()
    top = ""
    if search_result and search_result.get("results"):
        top = str(search_result["results"][0].get("memory", ""))
    top_is_new = bool(new_word) and new_word.lower() in top.lower() and (
        not old_word or old_word.lower() not in top.lower()
    )
    return Mem0ContradictionResult(profile=profile.name, stored_after_both_adds=stored,
                                   old_fact_still_present=old_present, new_fact_present=new_present,
                                   top_search_result=top, top_result_is_new_not_blended=top_is_new)


# =================================================================================================
# Orchestration
# =================================================================================================


def _argus_structural(statements: tuple[Statement, ...]) -> list[StructuralCase]:
    out = []
    for i, s in enumerate(statements):
        kinds = tuple(f.kind for f in extract(s.text))
        out.append(StructuralCase(
            text=s.text, gap=_structural_gap_name(s, i), argus_kinds_found=kinds,
            argus_facts_expected=len(s.mem0_expected), mem0_facts_recalled=0,
            mem0_facts_expected=len(s.mem0_expected), mem0_memories=(),
        ))
    return out


def main(*, max_workers: int = MAX_WORKERS, subprocess_timeout: float = 3600.0) -> dict[str, Any]:
    all_statements = (
        TUNING_STATEMENTS + TUNING_STRUCTURAL
        + HELD_OUT_STATEMENTS + HELD_OUT_STRUCTURAL
    )
    chains: list[dict[str, Any]] = []
    chains += build_statement_chains(TUNING_STATEMENTS, "tuning-stmt")
    chains += build_statement_chains(TUNING_STRUCTURAL, "tuning-struct")
    chains += build_sentence_chains(NOT_ABOUT_ME, "tuning-nam")
    chains += build_task_chains(TUNING_TASKS, "tuning-task")
    chains += build_statement_chains(HELD_OUT_STATEMENTS, "held-stmt")
    chains += build_statement_chains(HELD_OUT_STRUCTURAL, "held-struct")
    chains += build_sentence_chains(HELD_OUT_NOT_ABOUT_ME, "held-nam")
    chains += build_task_chains(HELD_OUT_TASKS, "held-task")
    chains += build_profile_chains((P1, P2, P3, P4))

    planned_adds = sum(1 for c in chains for op in c["ops"] if op["op"] == "add")

    mem0_response = mem0_loader.run_chains(chains, max_workers=max_workers,
                                           timeout=subprocess_timeout)
    results_by_id = {c["chain_id"]: c for c in mem0_response["chains"]}

    # --- ARGUS side ---
    argus_started = time.perf_counter()
    argus_tuning_recall = argus_recall(TUNING_STATEMENTS)
    argus_tuning_fp = argus_false_positives(NOT_ABOUT_ME)
    argus_held_recall = argus_recall(HELD_OUT_STATEMENTS)
    argus_held_fp = argus_false_positives(HELD_OUT_NOT_ABOUT_ME)
    argus_tuning_structural = _argus_structural(TUNING_STRUCTURAL)
    argus_held_structural = _argus_structural(HELD_OUT_STRUCTURAL)
    argus_extract_elapsed = time.perf_counter() - argus_started

    argus_tuning_contradiction = argus_contradiction(P3)
    argus_held_contradiction = argus_contradiction(P4)
    argus_tuning_flip = argus_mandate_flip(P3)
    argus_held_flip = argus_mandate_flip(P4)

    # --- mem0 side ---
    mem0_tuning_recall, tuning_statement_rows = _score_statement_chains(
        TUNING_STATEMENTS, results_by_id, "tuning-stmt")
    mem0_tuning_fp, tuning_nam_rows = _score_false_positive_chains(
        NOT_ABOUT_ME, results_by_id, "tuning-nam")
    mem0_held_recall, held_statement_rows = _score_statement_chains(
        HELD_OUT_STATEMENTS, results_by_id, "held-stmt")
    mem0_held_fp, held_nam_rows = _score_false_positive_chains(
        HELD_OUT_NOT_ABOUT_ME, results_by_id, "held-nam")

    def _score_structural(statements: tuple[Statement, ...], prefix: str,
                          argus_side: list[StructuralCase]) -> list[StructuralCase]:
        out = []
        for i, (s, argus_case) in enumerate(zip(statements, argus_side, strict=True)):
            chain = results_by_id[f"{prefix}-{i:04d}"]
            add_result = chain["results"][0] if chain.get("ok") and chain["results"] else None
            memories = [m["memory"] for m in (add_result or {}).get("memories", [])
                       if m.get("memory")]
            combined = " | ".join(memories)
            recalled = sum(1 for g in s.mem0_expected if _matches(combined, g))
            out.append(StructuralCase(
                text=s.text, gap=argus_case.gap, argus_kinds_found=argus_case.argus_kinds_found,
                argus_facts_expected=argus_case.argus_facts_expected,
                mem0_facts_recalled=recalled, mem0_facts_expected=len(s.mem0_expected),
                mem0_memories=tuple(memories),
            ))
        return out

    tuning_structural = _score_structural(
        TUNING_STRUCTURAL, "tuning-struct", argus_tuning_structural)
    held_structural = _score_structural(HELD_OUT_STRUCTURAL, "held-struct", argus_held_structural)

    mem0_tuning_tasks = _score_tasks(TUNING_TASKS,
                                     tuple((m,) for m in _tuning_task_keywords()),
                                     "tuning-task", results_by_id)
    mem0_held_tasks = _score_tasks(HELD_OUT_TASKS, _HELD_OUT_TASK_KEYWORDS, "held-task",
                                   results_by_id)

    argus_tuning_effect = _argus_effect(TUNING_TASKS)
    argus_held_effect = _argus_effect(HELD_OUT_TASKS)

    mem0_profile_p3 = score_mem0_profile(P3, results_by_id["profile-P3_contradiction"])
    mem0_profile_p4 = score_mem0_profile(P4, results_by_id["profile-P4_held_out_contradiction"])

    latency_cost = latency_and_cost(mem0_response, argus_extract_elapsed, len(all_statements))

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mem0_version": mem0_response.get("mem0_version"),
        "qwen_calls_planned": planned_adds,
        "qwen_calls_used": mem0_response["total_qwen_calls"],
        "qwen_total_tokens": mem0_response["total_usage_tokens"],
        "tuning": {
            "argus_recall": argus_tuning_recall.as_dict(),
            "mem0_recall": mem0_tuning_recall.as_dict(),
            "argus_false_positives": argus_tuning_fp.as_dict(),
            "mem0_false_positives": mem0_tuning_fp.as_dict(),
            "structural_cases": [c.as_dict() for c in tuning_structural],
            "argus_contradiction": argus_tuning_contradiction.as_dict(),
            "argus_mandate_flip": argus_tuning_flip.as_dict(),
            "mem0_contradiction": mem0_profile_p3.as_dict(),
            "argus_effect": argus_tuning_effect,
            "mem0_effect": mem0_tuning_tasks,
            "statement_rows": tuning_statement_rows,
            "not_about_me_rows": tuning_nam_rows,
        },
        "held_out": {
            "argus_recall": argus_held_recall.as_dict(),
            "mem0_recall": mem0_held_recall.as_dict(),
            "argus_false_positives": argus_held_fp.as_dict(),
            "mem0_false_positives": mem0_held_fp.as_dict(),
            "structural_cases": [c.as_dict() for c in held_structural],
            "argus_contradiction": argus_held_contradiction.as_dict(),
            "argus_mandate_flip": argus_held_flip.as_dict(),
            "mem0_contradiction": mem0_profile_p4.as_dict(),
            "argus_effect": argus_held_effect,
            "mem0_effect": mem0_held_tasks,
            "statement_rows": held_statement_rows,
            "not_about_me_rows": held_nam_rows,
        },
        "latency_and_cost": latency_cost,
    }
    write(REPORT_PATH, report)
    return report


# =================================================================================================
# held_out_2 — run separately from main(), against the ALREADY-WRITTEN artefact, so the settled
# `tuning`/`held_out_1` blocks are never re-spent against the Qwen budget.
# =================================================================================================


def _load_report() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        raise RuntimeError(f"{REPORT_PATH} does not exist yet — run main() first")
    import json

    return dict(json.loads(REPORT_PATH.read_text(encoding="utf-8")))


def freeze_held_out_2() -> dict[str, Any]:
    """Write `held_out_2`'s frozen texts (and `HELD_OUT_2_FROZEN_AT`) into the artefact, relabel
    the old `held_out` block to `held_out_1`, and save — before `extract()` or mem0 `add()` has
    touched a single `held_out_2` statement. Safe to call more than once (idempotent: relabelling
    is a no-op the second time, and the frozen texts are the same module constants every call)."""
    report = _load_report()
    if "held_out" in report and "held_out_1" not in report:
        old = report.pop("held_out")
        old["note"] = ("used for tuning after the 2026-09-30 lui/memory.py fixes — no longer "
                      "held-out")
        report["held_out_1"] = old
    report["held_out_2"] = {
        "frozen_at": HELD_OUT_2_FROZEN_AT,
        "note": "genuinely blind: written without opening lui/memory.py after the 2026-09-30 "
                "fixes; frozen here before extract() or mem0 add() touched any of it",
        "frozen_statements": [
            {"text": s.text, "argus_kind": s.argus_kind, "mem0_expected": list(s.mem0_expected)}
            for s in HELD_OUT_2_STATEMENTS
        ],
        "frozen_not_about_me": list(HELD_OUT_2_NOT_ABOUT_ME),
        "frozen_tasks": [
            {"statement": s, "question": q, "marker": m, "mem0_expected": list(kw)}
            for (s, q, m), kw in zip(HELD_OUT_2_TASKS, _HELD_OUT_2_TASK_KEYWORDS, strict=True)
        ],
        "frozen_structural": [
            {"text": c.text, "gap": c.gap, "mem0_expected": list(c.mem0_expected)}
            for c in HELD_OUT_2_STRUCTURAL
        ],
    }
    write(REPORT_PATH, report)
    return report


def _argus_structural_raw(cases: tuple[BlindStructuralCase, ...]) -> list[dict[str, Any]]:
    """Every real `Fact` `extract()` returns for each case, in full — kind, value, text and
    `replaces` — not a pass/fail boolean, per `BlindStructuralCase`'s own docstring."""
    out = []
    for case in cases:
        facts = extract(case.text)
        out.append({
            "text": case.text, "gap": case.gap,
            "facts": [{"kind": f.kind, "subject": f.subject, "value": f.value, "text": f.text,
                      "replaces": f.replaces} for f in facts],
        })
    return out


def main_held_out_2(
    *, max_workers: int = MAX_WORKERS, subprocess_timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run both real sides on `held_out_2`, merge the results into the existing artefact.

    Assumes :func:`freeze_held_out_2` already ran (raises if `held_out_2.frozen_at` is missing) —
    the frozen record must exist on disk before this spends a single Qwen call.

    **Two-phase, crash-resistant.** The real mem0 subprocess result is written to disk the moment
    it comes back — before a single ARGUS-side call runs — and the ARGUS-side calls that hit live
    product code (`_argus_effect`, which calls `argus.lui.server.handle_ask`) are individually
    guarded. Written after a real run lost 60 already-spent Qwen calls: `handle_ask` raised
    `AttributeError: module 'argus.lui.research.sizing' has no attribute 'loss_compare'` — a
    transient state in `lui/server.py` while it was being edited concurrently with this run (the
    very next read of the file showed a consistent `import ... as loss_sizing` at the call site) —
    and because the old code only wrote the artefact at the very end, the real mem0 results already
    paid for were never persisted. This version cannot lose already-spent Qwen calls to a
    downstream crash, product-code or otherwise."""
    report = _load_report()
    if "held_out_2" not in report or "frozen_at" not in report["held_out_2"]:
        raise RuntimeError("held_out_2 was not frozen yet — call freeze_held_out_2() first")

    chains: list[dict[str, Any]] = []
    chains += build_statement_chains(HELD_OUT_2_STATEMENTS, "held2-stmt")
    chains += build_statement_chains(
        tuple(Statement(c.text, None, c.mem0_expected) for c in HELD_OUT_2_STRUCTURAL),
        "held2-struct",
    )
    chains += build_sentence_chains(HELD_OUT_2_NOT_ABOUT_ME, "held2-nam")
    chains += build_task_chains(HELD_OUT_2_TASKS, "held2-task")
    planned_adds = sum(1 for c in chains for op in c["ops"] if op["op"] == "add")

    mem0_response = mem0_loader.run_chains(chains, max_workers=max_workers,
                                           timeout=subprocess_timeout)
    results_by_id = {c["chain_id"]: c for c in mem0_response["chains"]}

    # Phase 1: persist the real, already-paid-for mem0 result NOW, before anything that could crash.
    report["held_out_2"]["qwen_calls_planned_this_round"] = planned_adds
    report["held_out_2"]["qwen_calls_used_this_round"] = mem0_response["total_qwen_calls"]
    report["held_out_2"]["qwen_tokens_used_this_round"] = mem0_response["total_usage_tokens"]
    report["held_out_2"]["_raw_mem0_response"] = mem0_response
    report["qwen_calls_planned"] = report.get("qwen_calls_planned", 0) + planned_adds
    report["qwen_calls_used"] = report.get("qwen_calls_used", 0) + mem0_response["total_qwen_calls"]
    report["qwen_total_tokens"] = (
        report.get("qwen_total_tokens", 0) + mem0_response["total_usage_tokens"]
    )
    write(REPORT_PATH, report)

    # Phase 2: score both sides. Each ARGUS-side call that touches live product code is guarded —
    # a crash here must not un-happen the Qwen spend Phase 1 already persisted.
    try:
        argus_recall_score = argus_recall(HELD_OUT_2_STATEMENTS)
    except Exception as exc:
        argus_recall_score = RecallScore(total=len(HELD_OUT_2_STATEMENTS), recalled=0,
                                         missed=(f"ERROR: {type(exc).__name__}: {exc}",),
                                         elapsed_s=0.0)
    try:
        argus_fp_score = argus_false_positives(HELD_OUT_2_NOT_ABOUT_ME)
    except Exception as exc:
        argus_fp_score = FalsePositiveScore(
            total=len(HELD_OUT_2_NOT_ABOUT_ME), false_positives=0,
            examples=((f"ERROR: {type(exc).__name__}: {exc}", ()),), elapsed_s=0.0)
    mem0_recall_score, statement_rows = _score_statement_chains(
        HELD_OUT_2_STATEMENTS, results_by_id, "held2-stmt")
    mem0_fp_score, nam_rows = _score_false_positive_chains(
        HELD_OUT_2_NOT_ABOUT_ME, results_by_id, "held2-nam")

    try:
        argus_structural_raw = _argus_structural_raw(HELD_OUT_2_STRUCTURAL)
    except Exception as exc:
        argus_structural_raw = [
            {"text": c.text, "gap": c.gap, "facts": [],
             "error": f"{type(exc).__name__}: {exc}"} for c in HELD_OUT_2_STRUCTURAL
        ]
    mem0_structural = []
    for i, case in enumerate(HELD_OUT_2_STRUCTURAL):
        chain = results_by_id[f"held2-struct-{i:04d}"]
        add_result = chain["results"][0] if chain.get("ok") and chain["results"] else None
        memories = [m["memory"] for m in (add_result or {}).get("memories", [])
                   if m.get("memory")]
        combined = " | ".join(memories)
        recalled = sum(1 for g in case.mem0_expected if _matches(combined, g))
        mem0_structural.append({
            "text": case.text, "gap": case.gap, "mem0_memories": memories,
            "mem0_facts_recalled": recalled, "mem0_facts_expected": len(case.mem0_expected),
        })
    structural_combined = [
        {**argus_row, **{k: v for k, v in mem0_row.items() if k not in ("text", "gap")}}
        for argus_row, mem0_row in zip(argus_structural_raw, mem0_structural, strict=True)
    ]

    try:
        argus_effect_result = _argus_effect(HELD_OUT_2_TASKS)
    except Exception as exc:
        argus_effect_result = {"error": f"{type(exc).__name__}: {exc}",
                               "tasks": len(HELD_OUT_2_TASKS)}
    mem0_effect_result = _score_tasks(
        HELD_OUT_2_TASKS, _HELD_OUT_2_TASK_KEYWORDS, "held2-task", results_by_id)

    report = _load_report()  # re-read: Phase 1's write is the source of truth for the raw response
    report["held_out_2"].update({
        "argus_recall": argus_recall_score.as_dict(),
        "mem0_recall": mem0_recall_score.as_dict(),
        "argus_false_positives": argus_fp_score.as_dict(),
        "mem0_false_positives": mem0_fp_score.as_dict(),
        "structural_cases": structural_combined,
        "argus_effect": argus_effect_result,
        "mem0_effect": mem0_effect_result,
        "statement_rows": statement_rows,
        "not_about_me_rows": nam_rows,
    })
    del report["held_out_2"]["_raw_mem0_response"]  # only needed to survive a Phase-2 crash
    write(REPORT_PATH, report)
    return report


# =================================================================================================
# held_out_3 — ARGUS's side now spends real Qwen calls too (`memory_model.combined`), so this round
# is written for maximum crash-resistance: every single item, on either side, is persisted to the
# artefact the moment its result comes back, never batched behind a later step that could fail.
# =================================================================================================

QWEN_ENV_PATH = Path(__file__).resolve().parents[4] / ".secrets" / "qwen.env"
_QWEN_KEYS = ("BITGET_QWEN_API_KEY", "BITGET_QWEN_BASE_URL", "BITGET_QWEN_MODEL")


def _load_qwen_env_into_os_environ() -> None:
    """The three named keys, and only those, read from the one named file, loaded into
    `os.environ` and never printed, logged or returned — mirrors `mem0_runner._load_qwen_env`,
    written separately because that one runs in the isolated mem0 venv and this runs in the ARGUS
    venv, two different processes that cannot share a loaded value."""
    import os

    values: dict[str, str] = {}
    with open(QWEN_ENV_PATH, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in _QWEN_KEYS:
                values[key] = value.strip()
    missing = [k for k in _QWEN_KEYS if k not in values or not values[k]]
    if missing:
        raise RuntimeError(f"qwen.env is missing required key(s): {missing}")
    for key, value in values.items():
        os.environ[key] = value


def _held_out_3_items() -> list[tuple[str, str, str]]:
    """Every message that needs a `memory_model.combined()` call, `(item_id, text, category)`."""
    items: list[tuple[str, str, str]] = []
    for i, s in enumerate(HELD_OUT_3_STATEMENTS):
        items.append((f"h3-stmt-{i:04d}", s.text, "statement"))
    for i, t in enumerate(HELD_OUT_3_NOT_ABOUT_ME):
        items.append((f"h3-nam-{i:04d}", t, "not_about_me"))
    for i, c in enumerate(HELD_OUT_3_STRUCTURAL):
        items.append((f"h3-struct-{i:04d}", c.text, "structural"))
    for i, (stmt, _q, _m) in enumerate(HELD_OUT_3_TASKS):
        items.append((f"h3-task-{i:04d}", stmt, "task"))
    return items


def freeze_held_out_3() -> dict[str, Any]:
    """Write `held_out_3`'s frozen texts (and `HELD_OUT_3_FROZEN_AT`) into the artefact — before
    `memory_model.combined()` or mem0 `add()` has touched a single `held_out_3` statement. Safe to
    call more than once."""
    report = _load_report()
    report["held_out_3"] = {
        "frozen_at": HELD_OUT_3_FROZEN_AT,
        "note": "genuinely blind: written without opening lui/memory_model.py or lui/memory.py; "
                "frozen here before either side touched any of it",
        "frozen_statements": [
            {"text": s.text, "argus_kind": s.argus_kind, "mem0_expected": list(s.mem0_expected)}
            for s in HELD_OUT_3_STATEMENTS
        ],
        "frozen_not_about_me": list(HELD_OUT_3_NOT_ABOUT_ME),
        "frozen_tasks": [
            {"statement": s, "question": q, "marker": m, "mem0_expected": list(kw)}
            for (s, q, m), kw in zip(HELD_OUT_3_TASKS, _HELD_OUT_3_TASK_KEYWORDS, strict=True)
        ],
        "frozen_structural": [
            {"text": c.text, "gap": c.gap, "mem0_expected": list(c.mem0_expected)}
            for c in HELD_OUT_3_STRUCTURAL
        ],
    }
    write(REPORT_PATH, report)
    return report


def _fact_to_dict(f: Any) -> dict[str, Any]:
    return {"kind": f.kind, "subject": f.subject, "value": f.value, "text": f.text, "at": f.at,
           "price_at": f.price_at, "replaces": f.replaces}


def _dict_to_fact(d: dict[str, Any]) -> Any:
    from argus.lui.memory import Fact

    return Fact(kind=d["kind"], subject=d["subject"], value=d["value"], text=d["text"],
               at=d.get("at", ""), price_at=d.get("price_at"), replaces=d.get("replaces", ""))


def run_argus_combined_phase(report: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Run real `memory_model.combined()` on every `held_out_3` message, one at a time, writing
    the artefact after EACH item — so a crash never loses an already-spent Qwen call. Returns the
    updated report and the live client (so the caller can also read `client.calls`/`client.budget`
    for the effect phase without building a second one)."""
    from datetime import UTC, datetime

    from argus.lui.server import _model_for

    _load_qwen_env_into_os_environ()
    client: Any = _model_for("memcmp-held3", count=False)  # the Router, with its call counter
    if client is None:
        raise RuntimeError(
            "no Qwen client available from _model_for — either credentials failed to load or "
            "the visitor's hourly allowance is exhausted"
        )
    from argus.lui import memory_model

    now = datetime.now(UTC)
    raw: dict[str, Any] = report["held_out_3"].get("_argus_combined_raw", {})
    for item_id, text, _category in _held_out_3_items():
        if item_id in raw:
            continue
        calls_before = client.calls
        tokens_before = client.budget.spent if client.budget else 0
        try:
            facts = memory_model.combined(text, client, now, price_of=None)
            raw[item_id] = {
                "ok": True, "facts": [_fact_to_dict(f) for f in facts],
                "calls_delta": client.calls - calls_before,
                "tokens_delta": (client.budget.spent if client.budget else 0) - tokens_before,
            }
        except Exception as exc:  # a single item's crash must not lose every other item's spend
            raw[item_id] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "calls_delta": client.calls - calls_before}
        report["held_out_3"]["_argus_combined_raw"] = raw
        write(REPORT_PATH, report)
    return report, client


def _argus_combined_recall(statements: tuple[Statement, ...], raw: dict[str, Any],
                           prefix: str) -> tuple[RecallScore, list[dict[str, Any]]]:
    recalled = 0
    missed: list[str] = []
    rows: list[dict[str, Any]] = []
    for i, stmt in enumerate(statements):
        item = raw.get(f"{prefix}-{i:04d}", {"ok": False, "error": "not run"})
        facts = [_dict_to_fact(f) for f in item.get("facts", [])] if item.get("ok") else []
        kinds = [f.kind for f in facts]
        full = stmt.argus_kind in kinds
        if full:
            recalled += 1
        else:
            missed.append(stmt.text)
        rows.append({"text": stmt.text, "argus_kind": stmt.argus_kind,
                    "facts": [_fact_to_dict(f) for f in facts], "recalled": full,
                    "calls_delta": item.get("calls_delta", 0), "error": item.get("error")})
    return RecallScore(total=len(statements), recalled=recalled, missed=tuple(missed),
                       elapsed_s=0.0), rows


def _argus_combined_false_positives(
    sentences: tuple[str, ...], raw: dict[str, Any], prefix: str,
) -> tuple[FalsePositiveScore, list[dict[str, Any]]]:
    hits: list[tuple[str, tuple[str, ...]]] = []
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(sentences):
        item = raw.get(f"{prefix}-{i:04d}", {"ok": False, "error": "not run"})
        facts = [_dict_to_fact(f) for f in item.get("facts", [])] if item.get("ok") else []
        kinds = tuple(f.kind for f in facts)
        if kinds:
            hits.append((text, kinds))
        rows.append({"text": text, "facts": [_fact_to_dict(f) for f in facts],
                    "false_positive": bool(kinds), "calls_delta": item.get("calls_delta", 0),
                    "error": item.get("error")})
    return FalsePositiveScore(total=len(sentences), false_positives=len(hits),
                              examples=tuple(hits), elapsed_s=0.0), rows


def main_held_out_3(*, mem0_max_workers: int = MAX_WORKERS,
                    mem0_subprocess_timeout: float = 1800.0) -> dict[str, Any]:
    """Run both real sides on `held_out_3`. Assumes :func:`freeze_held_out_3` already ran.

    Order matters for crash-resistance: ARGUS's `combined()` phase runs first, one item at a time,
    persisted after each (see :func:`run_argus_combined_phase`); mem0's phase runs second and is
    persisted the moment its subprocess returns, before any scoring — the same two-phase discipline
    `main_held_out_2` learned the hard way. Safe to call again after a partial failure: both phases
    skip any item ID already present in the artefact.
    """
    report = _load_report()
    if "held_out_3" not in report or "frozen_at" not in report["held_out_3"]:
        raise RuntimeError("held_out_3 was not frozen yet — call freeze_held_out_3() first")

    # Phase 1: ARGUS's real combined() calls, one at a time, persisted after each.
    report, client = run_argus_combined_phase(report)
    argus_raw = report["held_out_3"]["_argus_combined_raw"]
    argus_calls_used = sum(item.get("calls_delta", 0) for item in argus_raw.values())
    argus_tokens_used = sum(item.get("tokens_delta", 0) for item in argus_raw.values())

    # Phase 2: mem0's real add() calls, one subprocess call, persisted immediately on return.
    mem0_structural_statements = tuple(
        Statement(c.text, None, c.mem0_expected) for c in HELD_OUT_3_STRUCTURAL
    )
    chains: list[dict[str, Any]] = []
    chains += build_statement_chains(HELD_OUT_3_STATEMENTS, "h3-stmt")
    chains += build_statement_chains(mem0_structural_statements, "h3-struct")
    chains += build_sentence_chains(HELD_OUT_3_NOT_ABOUT_ME, "h3-nam")
    chains += build_task_chains(HELD_OUT_3_TASKS, "h3-task")
    planned_mem0_adds = sum(1 for c in chains for op in c["ops"] if op["op"] == "add")
    # ARGUS's real per-item cost is unknowable before running it (a cheap pattern-gate may skip
    # the Qwen call entirely for a given message) — `len(_held_out_3_items())` is the same
    # "at most one call per message" ceiling the design promises, giving `qwen_calls_planned` a
    # genuine pre-spend upper bound rather than mixing in actual usage after the fact.
    planned_argus_calls = len(_held_out_3_items())

    remaining_budget = 150 - argus_calls_used
    if planned_mem0_adds > remaining_budget:
        raise RuntimeError(
            f"ARGUS's combined() phase spent {argus_calls_used} calls; mem0's planned "
            f"{planned_mem0_adds} would put this round over its 150-call budget "
            f"({argus_calls_used + planned_mem0_adds} > 150) — aborting before mem0 spends "
            f"anything. ARGUS's results are already persisted."
        )

    mem0_response = mem0_loader.run_chains(chains, max_workers=mem0_max_workers,
                                           timeout=mem0_subprocess_timeout)
    results_by_id = {c["chain_id"]: c for c in mem0_response["chains"]}
    report["held_out_3"]["qwen_calls_planned_this_round"] = planned_argus_calls + planned_mem0_adds
    report["held_out_3"]["argus_qwen_calls_used_this_round"] = argus_calls_used
    report["held_out_3"]["argus_qwen_tokens_used_this_round"] = argus_tokens_used
    report["held_out_3"]["mem0_qwen_calls_used_this_round"] = mem0_response["total_qwen_calls"]
    report["held_out_3"]["mem0_qwen_tokens_used_this_round"] = mem0_response["total_usage_tokens"]
    report["held_out_3"]["qwen_calls_used_this_round"] = (
        argus_calls_used + mem0_response["total_qwen_calls"]
    )
    report["held_out_3"]["_raw_mem0_response"] = mem0_response
    report["qwen_calls_planned"] = (
        report.get("qwen_calls_planned", 0) + planned_argus_calls + planned_mem0_adds
    )
    report["qwen_calls_used"] = (
        report.get("qwen_calls_used", 0) + argus_calls_used + mem0_response["total_qwen_calls"]
    )
    report["qwen_total_tokens"] = (
        report.get("qwen_total_tokens", 0) + argus_tokens_used + mem0_response["total_usage_tokens"]
    )
    write(REPORT_PATH, report)

    # Phase 3: score both sides from what is now safely on disk.
    argus_recall_score, statement_rows = _argus_combined_recall(
        HELD_OUT_3_STATEMENTS, argus_raw, "h3-stmt")
    argus_fp_score, nam_rows = _argus_combined_false_positives(
        HELD_OUT_3_NOT_ABOUT_ME, argus_raw, "h3-nam")
    mem0_recall_score, mem0_statement_rows = _score_statement_chains(
        HELD_OUT_3_STATEMENTS, results_by_id, "h3-stmt")
    mem0_fp_score, mem0_nam_rows = _score_false_positive_chains(
        HELD_OUT_3_NOT_ABOUT_ME, results_by_id, "h3-nam")

    structural_combined = []
    for i, case in enumerate(HELD_OUT_3_STRUCTURAL):
        argus_item = argus_raw.get(f"h3-struct-{i:04d}", {"ok": False})
        argus_facts = argus_item.get("facts", [])
        mem_chain = results_by_id[f"h3-struct-{i:04d}"]
        add_result = mem_chain["results"][0] if mem_chain.get("ok") and mem_chain["results"] \
            else None
        memories = [m["memory"] for m in (add_result or {}).get("memories", [])
                   if m.get("memory")]
        combined_text = " | ".join(memories)
        mem0_recalled = sum(1 for g in case.mem0_expected if _matches(combined_text, g))
        structural_combined.append({
            "text": case.text, "gap": case.gap, "argus_facts": argus_facts,
            "argus_error": argus_item.get("error"),
            "mem0_memories": memories, "mem0_facts_recalled": mem0_recalled,
            "mem0_facts_expected": len(case.mem0_expected),
        })

    # mem0's effect score is pure local scoring against Phase 2's already-persisted results — safe
    # regardless of what happens below, so it is computed unconditionally.
    mem0_effect_result = _score_tasks(
        HELD_OUT_3_TASKS, _HELD_OUT_3_TASK_KEYWORDS, "h3-task", results_by_id)
    try:
        argus_effect_result, effect_calls, effect_tokens = _held_out_3_effect(argus_raw, client)
    except Exception as exc:
        argus_effect_result = {"error": f"{type(exc).__name__}: {exc}",
                               "tasks": len(HELD_OUT_3_TASKS)}
        effect_calls = effect_tokens = 0

    report = _load_report()
    report["held_out_3"].update({
        "argus_recall": argus_recall_score.as_dict(),
        "mem0_recall": mem0_recall_score.as_dict(),
        "argus_false_positives": argus_fp_score.as_dict(),
        "mem0_false_positives": mem0_fp_score.as_dict(),
        "structural_cases": structural_combined,
        "argus_effect": argus_effect_result,
        "mem0_effect": mem0_effect_result,
        # `statement_rows`/`not_about_me_rows` stay mem0-side, matching every earlier round's
        # convention (`TestStatementRowsReconcileWithTheSummary` reconciles them against
        # `mem0_recall`/`mem0_false_positives`); the ARGUS side gets its own, differently-shaped
        # `argus_*` rows below rather than overloading the same key with a different meaning.
        "statement_rows": mem0_statement_rows,
        "not_about_me_rows": mem0_nam_rows,
        "argus_statement_rows": statement_rows,
        "argus_not_about_me_rows": nam_rows,
    })
    report["held_out_3"]["argus_qwen_calls_used_this_round"] += effect_calls
    report["held_out_3"]["argus_qwen_tokens_used_this_round"] += effect_tokens
    report["held_out_3"]["qwen_calls_used_this_round"] += effect_calls
    report["qwen_calls_used"] = report.get("qwen_calls_used", 0) + effect_calls
    report["qwen_total_tokens"] = report.get("qwen_total_tokens", 0) + effect_tokens
    del report["held_out_3"]["_argus_combined_raw"]
    del report["held_out_3"]["_raw_mem0_response"]
    write(REPORT_PATH, report)
    return report


def _held_out_3_effect(
    argus_raw: dict[str, Any], client: Any,
) -> tuple[dict[str, Any], int, int]:
    """The effect metric, built from Phase 1's ALREADY-EXTRACTED task facts rather than a second
    `combined()` call on the same statement text (which would double-spend a Qwen call this round
    can't spare) — `mem.dumps(mem.merge([], facts))` reproduces exactly the memory blob
    `handle_ask()` would have produced from that statement, then only the follow-up QUESTION goes
    through a real `handle_ask()` call (with the real client still live, in case routing ever needs
    it — the question texts here are the same proven patterns earlier rounds confirmed cost 0
    tokens, but this round measures rather than assumes that, via `client.calls` before/after)."""
    from argus.lui import memory as mem
    from argus.lui.server import handle_ask

    calls_before = client.calls
    tokens_before = client.budget.spent if client.budget else 0
    rows = []
    changed = 0
    leaked_or_ablation_carried = 0
    for i, (statement, question, marker) in enumerate(HELD_OUT_3_TASKS):
        item = argus_raw.get(f"h3-task-{i:04d}", {"ok": False, "facts": []})
        facts = [_dict_to_fact(f) for f in item.get("facts", [])] if item.get("ok") else []
        remembered = mem.dumps(mem.merge([], facts))
        with_memory = handle_ask(question, [], visitor="memcmp-h3-effect-a", memory=remembered)
        without = handle_ask(question, [], visitor="memcmp-h3-effect-a", memory="")
        other = handle_ask(question, [], visitor="memcmp-h3-effect-b", memory="")

        def carries(payload: dict[str, Any], marker: str = marker) -> bool:
            return any(str(line).startswith(marker) for line in payload.get("lines") or [])

        carried = carries(with_memory)
        fresh_carried = carries(other)
        if carried:
            changed += 1
        if carries(without) or fresh_carried:
            leaked_or_ablation_carried += 1
        rows.append({"statement": statement, "question": question, "marker": marker,
                    "with_memory_carries": carried, "without_memory_carries": carries(without),
                    "other_trader_carries": fresh_carried})
    argus_effect = {"tasks": len(rows), "changed_by_memory": changed,
                    "leaked_to_other_trader": sum(1 for r in rows if r["other_trader_carries"]),
                    "ablation_carried": sum(1 for r in rows if r["without_memory_carries"]),
                    "rows": rows}
    calls_used = client.calls - calls_before
    tokens_used = (client.budget.spent if client.budget else 0) - tokens_before
    return argus_effect, calls_used, tokens_used


# =================================================================================================
# held_out_4 — mirrors held_out_3's functions exactly (same two-phase, persist-then-score,
# crash-resistant discipline), written separately rather than generalised so nothing about the
# already-spent, already-validated held_out_3 run is touched.
# =================================================================================================


def _held_out_4_items() -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for i, s in enumerate(HELD_OUT_4_STATEMENTS):
        items.append((f"h4-stmt-{i:04d}", s.text, "statement"))
    for i, t in enumerate(HELD_OUT_4_NOT_ABOUT_ME):
        items.append((f"h4-nam-{i:04d}", t, "not_about_me"))
    for i, c in enumerate(HELD_OUT_4_STRUCTURAL):
        items.append((f"h4-struct-{i:04d}", c.text, "structural"))
    for i, (stmt, _q, _m) in enumerate(HELD_OUT_4_TASKS):
        items.append((f"h4-task-{i:04d}", stmt, "task"))
    return items


def freeze_held_out_4() -> dict[str, Any]:
    """Write `held_out_4`'s frozen texts into the artefact, and relabel `held_out_3` — its rows
    informed the second `lui/memory_model.py` change, so it is tuning data now, per the
    coordinator's exact instruction."""
    report = _load_report()
    if "held_out_3" in report and "used for tuning" not in report["held_out_3"].get("note", ""):
        report["held_out_3"]["note"] = (
            "held_out_3 (used for tuning after the second memory_model change) — its rows "
            "informed a real edit to lui/memory_model.py; no longer held-out"
        )
    report["held_out_4"] = {
        "frozen_at": HELD_OUT_4_FROZEN_AT,
        "note": "genuinely blind: written without opening lui/memory_model.py or lui/memory.py "
                "after the second memory_model change; frozen here before either side touched "
                "any of it",
        "frozen_statements": [
            {"text": s.text, "argus_kind": s.argus_kind, "mem0_expected": list(s.mem0_expected)}
            for s in HELD_OUT_4_STATEMENTS
        ],
        "frozen_not_about_me": list(HELD_OUT_4_NOT_ABOUT_ME),
        "frozen_tasks": [
            {"statement": s, "question": q, "marker": m, "mem0_expected": list(kw)}
            for (s, q, m), kw in zip(HELD_OUT_4_TASKS, _HELD_OUT_4_TASK_KEYWORDS, strict=True)
        ],
        "frozen_structural": [
            {"text": c.text, "gap": c.gap, "mem0_expected": list(c.mem0_expected)}
            for c in HELD_OUT_4_STRUCTURAL
        ],
    }
    write(REPORT_PATH, report)
    return report


def run_argus_combined_phase_4(report: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Same as :func:`run_argus_combined_phase`, targeting `held_out_4`'s items and visitor id —
    see that function's docstring for the crash-resistance rationale."""
    from datetime import UTC, datetime

    from argus.lui.server import _model_for

    _load_qwen_env_into_os_environ()
    client: Any = _model_for("memcmp-held4", count=False)  # the Router, with its call counter
    if client is None:
        raise RuntimeError(
            "no Qwen client available from _model_for — either credentials failed to load or "
            "the visitor's hourly allowance is exhausted"
        )
    from argus.lui import memory_model

    now = datetime.now(UTC)
    raw: dict[str, Any] = report["held_out_4"].get("_argus_combined_raw", {})
    for item_id, text, _category in _held_out_4_items():
        if item_id in raw:
            continue
        calls_before = client.calls
        tokens_before = client.budget.spent if client.budget else 0
        try:
            facts = memory_model.combined(text, client, now, price_of=None)
            raw[item_id] = {
                "ok": True, "facts": [_fact_to_dict(f) for f in facts],
                "calls_delta": client.calls - calls_before,
                "tokens_delta": (client.budget.spent if client.budget else 0) - tokens_before,
            }
        except Exception as exc:
            raw[item_id] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "calls_delta": client.calls - calls_before}
        report["held_out_4"]["_argus_combined_raw"] = raw
        write(REPORT_PATH, report)
    return report, client


def _held_out_4_effect(argus_raw: dict[str, Any], client: Any) -> tuple[dict[str, Any], int, int]:
    """Same as :func:`_held_out_3_effect`, targeting `held_out_4`'s tasks and item ids."""
    from argus.lui import memory as mem
    from argus.lui.server import handle_ask

    calls_before = client.calls
    tokens_before = client.budget.spent if client.budget else 0
    rows = []
    changed = 0
    leaked_or_ablation_carried = 0
    for i, (statement, question, marker) in enumerate(HELD_OUT_4_TASKS):
        item = argus_raw.get(f"h4-task-{i:04d}", {"ok": False, "facts": []})
        facts = [_dict_to_fact(f) for f in item.get("facts", [])] if item.get("ok") else []
        remembered = mem.dumps(mem.merge([], facts))
        with_memory = handle_ask(question, [], visitor="memcmp-h4-effect-a", memory=remembered)
        without = handle_ask(question, [], visitor="memcmp-h4-effect-a", memory="")
        other = handle_ask(question, [], visitor="memcmp-h4-effect-b", memory="")

        def carries(payload: dict[str, Any], marker: str = marker) -> bool:
            return any(str(line).startswith(marker) for line in payload.get("lines") or [])

        carried = carries(with_memory)
        fresh_carried = carries(other)
        if carried:
            changed += 1
        if carries(without) or fresh_carried:
            leaked_or_ablation_carried += 1
        rows.append({"statement": statement, "question": question, "marker": marker,
                    "with_memory_carries": carried, "without_memory_carries": carries(without),
                    "other_trader_carries": fresh_carried})
    argus_effect = {"tasks": len(rows), "changed_by_memory": changed,
                    "leaked_to_other_trader": sum(1 for r in rows if r["other_trader_carries"]),
                    "ablation_carried": sum(1 for r in rows if r["without_memory_carries"]),
                    "rows": rows}
    calls_used = client.calls - calls_before
    tokens_used = (client.budget.spent if client.budget else 0) - tokens_before
    return argus_effect, calls_used, tokens_used


def main_held_out_4(
    *, mem0_max_workers: int = MAX_WORKERS, mem0_subprocess_timeout: float = 1800.0,
    budget_cap: int = 110,
) -> dict[str, Any]:
    """Same two-phase, crash-resistant shape as :func:`main_held_out_3`, targeting `held_out_4`.
    ``budget_cap`` is this round's own ceiling (110, per the coordinator's instruction) — enforced
    before mem0 spends anything, the same way `main_held_out_3` enforces its own 150."""
    report = _load_report()
    if "held_out_4" not in report or "frozen_at" not in report["held_out_4"]:
        raise RuntimeError("held_out_4 was not frozen yet — call freeze_held_out_4() first")

    report, client = run_argus_combined_phase_4(report)
    argus_raw = report["held_out_4"]["_argus_combined_raw"]
    argus_calls_used = sum(item.get("calls_delta", 0) for item in argus_raw.values())
    argus_tokens_used = sum(item.get("tokens_delta", 0) for item in argus_raw.values())

    mem0_structural_statements = tuple(
        Statement(c.text, None, c.mem0_expected) for c in HELD_OUT_4_STRUCTURAL
    )
    chains: list[dict[str, Any]] = []
    chains += build_statement_chains(HELD_OUT_4_STATEMENTS, "h4-stmt")
    chains += build_statement_chains(mem0_structural_statements, "h4-struct")
    chains += build_sentence_chains(HELD_OUT_4_NOT_ABOUT_ME, "h4-nam")
    chains += build_task_chains(HELD_OUT_4_TASKS, "h4-task")
    planned_mem0_adds = sum(1 for c in chains for op in c["ops"] if op["op"] == "add")
    planned_argus_calls = len(_held_out_4_items())

    remaining_budget = budget_cap - argus_calls_used
    if planned_mem0_adds > remaining_budget:
        raise RuntimeError(
            f"ARGUS's combined() phase spent {argus_calls_used} calls; mem0's planned "
            f"{planned_mem0_adds} would put this round over its {budget_cap}-call budget "
            f"({argus_calls_used + planned_mem0_adds} > {budget_cap}) — aborting before mem0 "
            f"spends anything. ARGUS's results are already persisted."
        )

    mem0_response = mem0_loader.run_chains(chains, max_workers=mem0_max_workers,
                                           timeout=mem0_subprocess_timeout)
    results_by_id = {c["chain_id"]: c for c in mem0_response["chains"]}
    report["held_out_4"]["qwen_calls_planned_this_round"] = planned_argus_calls + planned_mem0_adds
    report["held_out_4"]["argus_qwen_calls_used_this_round"] = argus_calls_used
    report["held_out_4"]["argus_qwen_tokens_used_this_round"] = argus_tokens_used
    report["held_out_4"]["mem0_qwen_calls_used_this_round"] = mem0_response["total_qwen_calls"]
    report["held_out_4"]["mem0_qwen_tokens_used_this_round"] = mem0_response["total_usage_tokens"]
    report["held_out_4"]["qwen_calls_used_this_round"] = (
        argus_calls_used + mem0_response["total_qwen_calls"]
    )
    report["held_out_4"]["_raw_mem0_response"] = mem0_response
    report["qwen_calls_planned"] = (
        report.get("qwen_calls_planned", 0) + planned_argus_calls + planned_mem0_adds
    )
    report["qwen_calls_used"] = (
        report.get("qwen_calls_used", 0) + argus_calls_used + mem0_response["total_qwen_calls"]
    )
    report["qwen_total_tokens"] = (
        report.get("qwen_total_tokens", 0) + argus_tokens_used + mem0_response["total_usage_tokens"]
    )
    write(REPORT_PATH, report)

    argus_recall_score, statement_rows = _argus_combined_recall(
        HELD_OUT_4_STATEMENTS, argus_raw, "h4-stmt")
    argus_fp_score, nam_rows = _argus_combined_false_positives(
        HELD_OUT_4_NOT_ABOUT_ME, argus_raw, "h4-nam")
    mem0_recall_score, mem0_statement_rows = _score_statement_chains(
        HELD_OUT_4_STATEMENTS, results_by_id, "h4-stmt")
    mem0_fp_score, mem0_nam_rows = _score_false_positive_chains(
        HELD_OUT_4_NOT_ABOUT_ME, results_by_id, "h4-nam")

    structural_combined = []
    for i, case in enumerate(HELD_OUT_4_STRUCTURAL):
        argus_item = argus_raw.get(f"h4-struct-{i:04d}", {"ok": False})
        argus_facts = argus_item.get("facts", [])
        mem_chain = results_by_id[f"h4-struct-{i:04d}"]
        add_result = mem_chain["results"][0] if mem_chain.get("ok") and mem_chain["results"] \
            else None
        memories = [m["memory"] for m in (add_result or {}).get("memories", [])
                   if m.get("memory")]
        combined_text = " | ".join(memories)
        mem0_recalled = sum(1 for g in case.mem0_expected if _matches(combined_text, g))
        structural_combined.append({
            "text": case.text, "gap": case.gap, "argus_facts": argus_facts,
            "argus_error": argus_item.get("error"),
            "mem0_memories": memories, "mem0_facts_recalled": mem0_recalled,
            "mem0_facts_expected": len(case.mem0_expected),
        })

    mem0_effect_result = _score_tasks(
        HELD_OUT_4_TASKS, _HELD_OUT_4_TASK_KEYWORDS, "h4-task", results_by_id)
    try:
        argus_effect_result, effect_calls, effect_tokens = _held_out_4_effect(argus_raw, client)
    except Exception as exc:
        argus_effect_result = {"error": f"{type(exc).__name__}: {exc}",
                               "tasks": len(HELD_OUT_4_TASKS)}
        effect_calls = effect_tokens = 0

    report = _load_report()
    report["held_out_4"].update({
        "argus_recall": argus_recall_score.as_dict(),
        "mem0_recall": mem0_recall_score.as_dict(),
        "argus_false_positives": argus_fp_score.as_dict(),
        "mem0_false_positives": mem0_fp_score.as_dict(),
        "structural_cases": structural_combined,
        "argus_effect": argus_effect_result,
        "mem0_effect": mem0_effect_result,
        "statement_rows": mem0_statement_rows,
        "not_about_me_rows": mem0_nam_rows,
        "argus_statement_rows": statement_rows,
        "argus_not_about_me_rows": nam_rows,
    })
    report["held_out_4"]["argus_qwen_calls_used_this_round"] += effect_calls
    report["held_out_4"]["argus_qwen_tokens_used_this_round"] += effect_tokens
    report["held_out_4"]["qwen_calls_used_this_round"] += effect_calls
    report["qwen_calls_used"] = report.get("qwen_calls_used", 0) + effect_calls
    report["qwen_total_tokens"] = report.get("qwen_total_tokens", 0) + effect_tokens
    del report["held_out_4"]["_argus_combined_raw"]
    del report["held_out_4"]["_raw_mem0_response"]
    write(REPORT_PATH, report)
    return report


def _tuning_task_keywords() -> tuple[str, ...]:
    """One representative keyword per `TUNING_TASKS` row, derived from the SAME hand-written
    keyword already assigned to that exact statement text in `TUNING_STATEMENTS`/tuning fixtures —
    not a second independent judgement call, just a lookup by statement text."""
    lookup: dict[str, str] = {}
    for s in TUNING_STATEMENTS:
        if s.mem0_expected:
            lookup[s.text] = s.mem0_expected[0][0]
    # A few TUNING_TASKS statements are paraphrases already covered above by exact text match;
    # the remainder are hand-covered here directly.
    manual = {
        "I can't lose more than 10%": "10", "I can't lose more than 5%": "5",
        "my max drawdown is 15%": "15", "my loss limit is 8%": "8",
        "I never lose more than 20% on anything": "20",
        "I'm a swing trader": "swing", "Im a day trader": "day",
        "I usually hold for weeks": "week", "I hold for months": "month",
        "my risk budget is 20%": "20", "no single name above 30% of my risk": "30",
    }
    out = []
    for statement, _q, _m in TUNING_TASKS:
        out.append(lookup.get(statement) or manual.get(statement) or statement.split()[0])
    return tuple(out)


def _argus_effect(tasks: tuple[tuple[str, str, str], ...]) -> dict[str, Any]:
    """Real `handle_ask()`, exactly `memory_eval.effect()`'s own method (`memory_eval.py:132-161`),
    reimplemented here (not imported) only because that function is hardcoded to `memory_eval.TASKS`
    and this needs it to run over `HELD_OUT_TASKS` too; the body is unchanged."""
    import os

    for key in list(os.environ):
        if "QWEN" in key:
            del os.environ[key]
    from argus.lui.server import handle_ask

    rows = []
    for statement, question, marker in tasks:
        first = handle_ask(statement, [], visitor="memcmp-effect-a")
        remembered = str(first.get("memory") or "")
        with_memory = handle_ask(question, [], visitor="memcmp-effect-a", memory=remembered)
        without = handle_ask(question, [], visitor="memcmp-effect-a", memory="")
        other = handle_ask(question, [], visitor="memcmp-effect-b", memory="")

        def carries(payload: dict[str, Any], marker: str = marker) -> bool:
            return any(str(line).startswith(marker) for line in payload.get("lines") or [])

        rows.append({"statement": statement, "question": question, "marker": marker,
                    "with_memory_carries": carries(with_memory),
                    "without_memory_carries": carries(without),
                    "other_trader_carries": carries(other)})
    changed = sum(1 for r in rows if r["with_memory_carries"] and not r["without_memory_carries"])
    return {"tasks": len(rows), "changed_by_memory": changed,
           "leaked_to_other_trader": sum(1 for r in rows if r["other_trader_carries"]),
           "ablation_carried": sum(1 for r in rows if r["without_memory_carries"]), "rows": rows}


def latency_and_cost(mem0_response: dict[str, Any], argus_extract_elapsed_s: float,
                     argus_statement_count: int) -> dict[str, Any]:
    """ARGUS: total wall time over every `extract()` call in the statement/false-positive sets,
    divided by call count — pure regex, no network, so this number is dominated by Python function-
    call overhead, not any real work. mem0: mean/median/p95 of each real `add()`'s own `elapsed_s`
    (timed inside `mem0_runner.py`, around `Memory.add()` only — NOT including that worker's one-
    time `Memory()` construction cost, which this function does not report at all, since a
    long-running deployment pays it once, not per turn, and folding it into a per-call number
    would overstate mem0's real per-turn cost by an amount that shrinks with volume; see the
    `"note"` field this function returns for exactly what is and is not included)."""
    add_elapsed: list[float] = []
    add_tokens: list[int] = []
    search_elapsed: list[float] = []
    for chain in mem0_response["chains"]:
        for op in chain.get("results", []):
            if op.get("op") == "add" and op.get("ok"):
                add_elapsed.append(op["elapsed_s"])
                usage = op.get("usage") or {}
                if usage.get("total_tokens") is not None:
                    add_tokens.append(usage["total_tokens"])
            elif op.get("op") in ("search", "get_all") and op.get("ok"):
                search_elapsed.append(op["elapsed_s"])

    def _stats(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(xs)
        n = len(s)
        return {"mean": sum(s) / n, "median": s[n // 2], "p95": s[min(n - 1, int(n * 0.95))],
               "min": s[0], "max": s[-1]}

    return {
        "argus": {
            "extract_calls": argus_statement_count,
            "total_elapsed_s": round(argus_extract_elapsed_s, 6),
            "mean_elapsed_ms": round(1000 * argus_extract_elapsed_s / argus_statement_count, 6)
            if argus_statement_count else 0.0,
            "qwen_calls_per_turn": 0, "qwen_tokens_per_turn": 0,
        },
        "mem0": {
            "add_calls": len(add_elapsed),
            "add_latency_s": {k: round(v, 3) for k, v in _stats(add_elapsed).items()},
            "search_or_get_all_calls": len(search_elapsed),
            "search_latency_s": {k: round(v, 6) for k, v in _stats(search_elapsed).items()},
            "qwen_calls_per_turn": 1.0,
            "tokens_per_add_call": {k: round(v, 1) for k, v in
                                    _stats([float(t) for t in add_tokens]).items()},
            "note": "add_latency_s excludes each worker's one-time Memory() construction "
                    "(sentence-transformers/qdrant/spaCy load, ~15-55s, paid once per worker "
                    "thread, not per turn — not itself timed as a per-call cost here); "
                    "search()/get_all() cost zero Qwen calls (confirmed by reading mem0/memory/"
                    "main.py: neither method's body calls self.llm anywhere).",
        },
    }


def render(report: dict[str, Any]) -> str:
    lines = ["MEMORY COMPARISON — ARGUS lui.memory vs. real mem0ai 2.2.1", ""]
    lines.append(f"mem0 version: {report['mem0_version']}  qwen calls used: "
                f"{report['qwen_calls_used']}/{report['qwen_calls_planned']} planned  "
                f"total tokens: {report['qwen_total_tokens']}")
    for split in ("tuning", "held_out"):
        d = report[split]
        ar, mr = d["argus_recall"], d["mem0_recall"]
        af, mf = d["argus_false_positives"], d["mem0_false_positives"]
        lines.append(f"\n[{split}]")
        lines.append(f"  recall  ARGUS {ar['recalled']}/{ar['total']} ({ar['recall_rate']:.1%})  "
                    f"mem0 {mr['recalled']}/{mr['total']} ({mr['recall_rate']:.1%})")
        lines.append(f"  false+  ARGUS {af['false_positives']}/{af['total']}  "
                    f"mem0 {mf['false_positives']}/{mf['total']}")
        ac = d["argus_contradiction"]
        lines.append(f"  contradiction  ARGUS latest-wins-all={ac['all_latest_win']} "
                    f"replaces-recorded-all={ac['all_replaces_recorded']}  "
                    f"flip={d['argus_mandate_flip']['verdict_flipped']}")
        mc = d["mem0_contradiction"]
        lines.append(f"  mem0 contradiction: old_still_present={mc['old_fact_still_present']} "
                    f"new_present={mc['new_fact_present']} "
                    f"top_result_is_new_not_blended={mc['top_result_is_new_not_blended']}")
        ef, me = d["argus_effect"], d["mem0_effect"]
        lines.append(f"  effect  ARGUS changed {ef['changed_by_memory']}/{ef['tasks']}, "
                    f"leaked {ef['leaked_to_other_trader']}  "
                    f"mem0 changed {me['changed_by_memory']}/{me['tasks']}, "
                    f"fresh-user-never-carries={me['fresh_user_never_carries']}")
    lc = report["latency_and_cost"]
    lines.append(f"\nlatency: ARGUS mean {lc['argus']['mean_elapsed_ms']:.4f}ms/extract  "
                f"mem0 add() mean {lc['mem0']['add_latency_s']['mean']:.2f}s, "
                f"p95 {lc['mem0']['add_latency_s']['p95']:.2f}s")
    lines.append(f"cost: ARGUS 0 tokens/turn always  "
                f"mem0 mean {lc['mem0']['tokens_per_add_call']['mean']:.0f} tokens/add() call")
    return "\n".join(lines)


if __name__ == "__main__":
    result = main()
    print(render(result))
    print(f"\nsaved -> {REPORT_PATH}")


__all__ = [
    "HELD_OUT_2_FROZEN_AT",
    "HELD_OUT_2_NOT_ABOUT_ME",
    "HELD_OUT_2_STATEMENTS",
    "HELD_OUT_2_STRUCTURAL",
    "HELD_OUT_2_TASKS",
    "HELD_OUT_3_FROZEN_AT",
    "HELD_OUT_3_NOT_ABOUT_ME",
    "HELD_OUT_3_STATEMENTS",
    "HELD_OUT_3_STRUCTURAL",
    "HELD_OUT_3_TASKS",
    "HELD_OUT_4_FROZEN_AT",
    "HELD_OUT_4_NOT_ABOUT_ME",
    "HELD_OUT_4_STATEMENTS",
    "HELD_OUT_4_STRUCTURAL",
    "HELD_OUT_4_TASKS",
    "HELD_OUT_NOT_ABOUT_ME",
    "HELD_OUT_STATEMENTS",
    "HELD_OUT_STRUCTURAL",
    "HELD_OUT_TASKS",
    "NOT_ABOUT_ME",
    "P1",
    "P2",
    "P3",
    "P4",
    "REPORT_PATH",
    "TUNING_STATEMENTS",
    "TUNING_STRUCTURAL",
    "TUNING_TASKS",
    "BlindStructuralCase",
    "Statement",
    "argus_contradiction",
    "argus_false_positives",
    "argus_mandate_flip",
    "argus_recall",
    "build_profile_chains",
    "build_sentence_chains",
    "build_statement_chains",
    "build_task_chains",
    "freeze_held_out_2",
    "freeze_held_out_3",
    "freeze_held_out_4",
    "latency_and_cost",
    "main",
    "main_held_out_2",
    "main_held_out_3",
    "main_held_out_4",
    "render",
    "run_argus_combined_phase",
    "run_argus_combined_phase_4",
    "score_mem0_profile",
]
