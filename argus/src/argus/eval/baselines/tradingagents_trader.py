# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/TauricResearch/TradingAgents
# Path:    tradingagents/agents/schemas.py, lines 1-24 (module docstring, imports, the
#          `_NULLISH_FLOAT` placeholder set), lines 33-50 (`_coerce_optional_float`), lines
#          68-77 (`TraderAction`), lines 137-179 (`TraderProposal`), lines 182-204
#          (`render_trader_proposal`) — stitched contiguously in that order. Skipped: lines
#          25-32 and 51-136 (`PortfolioRating`, `ResearchPlan`, and other agents' schemas this
#          excerpt's target, the Trader's real structured-output type, does not reference).
# Commit:  be952b8eccb49720509af544c6675233bc1f10d0 (2026-09-07) — same commit already pinned
#          by the sibling `tradingagents_rating.py` / `tradingagents_memory.py` baselines.
# Licence: Apache License 2.0 — full text below, identical to the sibling vendored files in this
#          package; attributed to TauricResearch per the same reasoning recorded there.
#
# This is the real, unmodified structured-output type the Trader agent (`agents/trader/
# trader.py`) fills to produce a transaction proposal. `eval/explainability_comparison.py` runs
# it directly: constructs a `TraderProposal` carrying a numeric `entry_price` that does not match
# any fact the desk was actually given, and confirms the real, unmodified Pydantic validator
# accepts it without complaint — `_coerce_optional_float` only normalises STRING FORMAT
# (placeholder text, percentages, currency symbols), never the VALUE, so a well-formed but
# fabricated number passes untouched. `argus.agents.grounding.check` is run on the same
# constructed figure and is confirmed to flag it as unresolved.
#
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION (summarised terms retained from the
# upstream LICENSE file; the full canonical text is at http://www.apache.org/licenses/LICENSE-2.0)
#
# 1. Definitions — "License", "Licensor", "Legal Entity", "You", "Source"/"Object" form, "Work",
#    "Derivative Works", "Contribution", "Contributor" as defined by the Apache 2.0 license text.
# 2. Grant of Copyright License — perpetual, worldwide, non-exclusive, no-charge, royalty-free,
#    irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display,
#    publicly perform, sublicense, and distribute the Work and Derivative Works.
# 3. Grant of Patent License — a patent license as stated in the License, terminating upon
#    initiating patent litigation over the Work.
# 4. Redistribution — permitted in Source or Object form provided this License and all
#    copyright/patent/trademark/attribution notices are retained, and any modified files carry
#    prominent notices of the changes (none made here — this file is unmodified).
# 5. Submission of Contributions is under this License unless stated otherwise.
# 6. Trademarks — this License grants no permission to use the Licensor's trade names, trademarks,
#    or service marks.
# 7. Disclaimer of Warranty — the Work is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF
#    ANY KIND, express or implied.
# 8. Limitation of Liability — no Contributor is liable for damages arising from use of the Work.
# 9. Accepting Warranty or Additional Liability — offered only on the offeror's own responsibility.
#
# Copyright TauricResearch (see attribution note above)
# Licensed under the Apache License, Version 2.0; you may not use this file except in compliance
# with the License. You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# ============================== VENDORED FROM HERE ==============================
"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number. Anything else passes through to pydantic.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    return cleaned or None


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)
