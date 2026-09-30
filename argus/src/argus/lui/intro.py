"""What this console is, who it is for, and what to ask it: the first question a stranger asks.

"What is this site and who is it for" was answered '"that" has nothing to refer to yet', and "what
is ARGUS" and "what can you do" were declined (a first-time user, 2026-09-30). The product's own
description was on the README and nowhere a visitor could ask for it. The standing counts are read
from the register at the moment of asking, so this answer cannot drift from /proof.
"""

from __future__ import annotations

import re
from typing import Any

from argus.lui.provenance import explains
from argus.lui.trace import trace_module
from argus.truth.source import Source

_SUBJECT = (r"(?:argus|this(?:\s+(?:site|website|page|tool|app|console|product|thing|desk|"
            r"service|project))?)")

INTRO_Q = re.compile(
    rf"^\s*(?:(?:hi|hello|hey)\W+)?(?:so\s+)?(?:what\s+(?:is|'s)\s+{_SUBJECT}|what\s+does\s+"
    rf"{_SUBJECT}\s+do|who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|who\s+(?:made|built)\s+{_SUBJECT}|"
    rf"what\s+can\s+(?:you|argus|this(?:\s+\w+)?)\s+do(?:\s+for\s+me)?|what\s+(?:can|should)\s+i\s+"
    rf"ask(?:\s+(?:you|it|argus|here))?|how\s+do\s+i\s+use\s+{_SUBJECT}|what\s+am\s+i\s+looking\s+at|"
    rf"help|getting\s+started|where\s+do\s+i\s+start)"
    rf"(?:\s*(?:,|and)\s*(?:who\s+(?:is|'s)\s+(?:{_SUBJECT}|it)\s+for|what\s+(?:does\s+{_SUBJECT}\s+do|"
    rf"can\s+(?:you|it)\s+do)))?\s*[?.!]*\s*$",
    re.I)
"""A question about the product itself, asked whole: "what is this site and who is it for"."""


def answer() -> tuple[list[str], list[Source], dict[str, Any]]:
    from argus.eval.standing import REGISTER, State

    counts = {s: sum(1 for cap in REGISTER if cap.state is s) for s in State}
    total = len(REGISTER)
    lines = [
        "Bottom line: ARGUS is a research console for Bitget markets — tokenized US stocks, "
        "crypto, indices and commodities. Ask in plain words and it works the answer out from "
        "live data, showing where every number came from.",
        "Who it is for: a trader holding Bitget's tokenized US stocks who wants a second desk "
        "that shows its work. It is not a signal service and not financial advice, and it never "
        "places an order for you.",
        "What to ask, for example: \"what is NVDA doing today\", \"how much would I lose if the "
        "Nasdaq fell 10% and I hold $5k of TSLA\", \"size a trade: $10k account, 1% risk, stop "
        "4% below entry\", \"compare gold and bitcoin this year\", \"when does TSLA report and "
        "how big is the move usually\", or \"I have $1,000, where do I start\".",
        f"How far to trust it: every capability is measured against a named rival on the same "
        f"input — of {total}, {counts[State.OWNED]} win, {counts[State.TIED]} tie and "
        f"{counts[State.LOST]} lose, all listed on /proof — and what it got wrong stays on "
        f"/wrong. Ask \"how is your desk doing\" for its own paper record; the team's separate "
        f"trading agent, which places orders on Bitget's demo venue, is on /agent.",
    ]
    explains(*lines)
    return lines, [Source("computation", "argus.eval.standing register",
                          "the capability counts, read now")], {
        "standing": {s.value: n for s, n in counts.items()}, "total": total}


trace_module(globals())
