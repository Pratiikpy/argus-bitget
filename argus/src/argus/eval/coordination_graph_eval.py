"""Does a persistent account graph see coordination the per-story screen cannot? Measured on the
crowd posts the Track 2 sentiment agent recorded in its first paper run.

The kill criterion is the study's own (research/harvest/11-coortweet.md): if the graph catches
nothing the stateless per-story flag (`truth/novelty` / the agent's port of it) has not already
caught, it is dropped rather than shipped as a second signal that always agrees.

Input: every ``snapshot`` event of run 1's published ledger (`public/ledger.jsonl`, read only),
whose crowd section lists each story cluster with the accounts that carried it and whether the
per-story screen called it coordinated. A cluster seen in several snapshots is one story; its
accounts are the union. The snapshots record accounts per story but not how many posts each made,
so every share counts once and the symmetry score is 1 by construction here — reported, not used.

Accounts appear in the published report under a pseudonym (:func:`pseudonym`): a stable hash of
the handle, so the groups and the pairs are reproducible from the same record, but the report does
not put a named person's account beside the word "coordinated". The per-story screen is a
heuristic; a pair of accounts repeating together is evidence about a pattern, not a finding about
the people behind it.

    python -m argus.eval.coordination_graph_eval --record C:/path/to/run1/public
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from argus.truth.artefact import write
from argus.truth.coordination_graph import Share, components, edges, percentile_cut, strong
from argus.truth.paths import DATA_DIR

OUT = DATA_DIR / "coordination_graph_eval.json"


def stories(ledger: Path) -> tuple[dict[str, set[str]], dict[str, bool], int, int]:
    """Accounts per story, the per-story coordinated flag, snapshots read, crowd items seen."""
    accounts: dict[str, set[str]] = defaultdict(set)
    flagged: dict[str, bool] = {}
    snapshots = items = 0
    # Split on newline only: str.splitlines also breaks on U+2028 inside a post's text.
    for line in ledger.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") != "snapshot":
            continue
        snapshots += 1
        crowd = (event.get("payload", {}).get("snapshot") or {}).get("crowd") or {}
        items += int(crowd.get("items", 0))
        for c in crowd.get("clusters", []):
            story = str(c.get("cluster_id"))
            accounts[story].update(str(s) for s in c.get("sources", []) or [])
            flagged[story] = flagged.get(story, False) or bool(c.get("coordinated"))
    return accounts, flagged, snapshots, items


def pseudonym(account: str) -> str:
    """``x:`` or ``reddit:`` and the first 10 hex characters of the handle's SHA-256."""
    platform = "reddit" if account.startswith("u/") else "x"
    return f"{platform}:{hashlib.sha256(account.encode('utf-8')).hexdigest()[:10]}"


def _anonymous(edge: dict[str, Any]) -> dict[str, Any]:
    return {**edge, "left": pseudonym(str(edge["left"])), "right": pseudonym(str(edge["right"]))}


def run(record: Path) -> dict[str, Any]:
    accounts, flagged, snapshots, items = stories(record / "ledger.jsonl")
    shares = [Share(a, story) for story, group in accounts.items() for a in group]
    graph = edges(shares)
    kept = strong(graph)
    new = [e for e in kept if not any(flagged.get(s) for s in e.stories)]
    groups = components(kept)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": {"record": record.name, "snapshots": snapshots, "crowd_items": items,
                  "stories": len(accounts), "accounts": len({a for g in accounts.values()
                                                            for a in g}),
                  "stories_flagged_per_story": sum(flagged.values())},
        "graph": {"edges": len(graph), "percentile_cut": percentile_cut(graph),
                  "strong_edges": len(kept),
                  "groups": [sorted(pseudonym(a) for a in g) for g in groups[:20]],
                  "group_sizes": [len(g) for g in groups]},
        "incremental": {
            "strong_edges_on_no_flagged_story": len(new),
            "examples": [_anonymous(e.as_dict()) for e in new[:20]],
        },
        "verdict": ("the graph finds account pairs repeating together on stories the per-story "
                    "screen never flagged" if new else
                    "every strong edge sits on a story the per-story screen already flagged: no "
                    "incremental catch, so the graph is not shipped as a signal"),
        "caveats": ["share counts per account are not recorded in the snapshots, so symmetry is "
                    "1 for every edge and is not used",
                    "stories carried by more than 60 accounts add no pairs (edges(): "
                    "max_story_accounts)"],
    }


def main() -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    blob = run(args.record)
    write(OUT, blob)
    print(json.dumps({k: blob[k] for k in ("input", "incremental", "verdict")}, indent=1)[:3000])
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
