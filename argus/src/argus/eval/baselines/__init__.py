"""Verbatim third-party baselines, vendored for reproducible same-input comparison.

Standing Rule #3 (project ``CLAUDE.md``) requires reading the best implementation before writing
our own, and ``eval/standing.py``'s ``baseline_reproduced`` condition requires the comparison to
run the baseline, not a paraphrase of it. A description of a competitor's behaviour is a claim;
running the competitor's own code on the same input is evidence.

Each module in this package is an unmodified copy of one file from a specific commit of an MIT- or
similarly permissively-licensed external repository, kept byte-for-byte identical below its
provenance header so a diff against the upstream file at the cited commit shows zero changes to any
line of logic. ``loader.py`` handles making the vendored file importable (its own internal imports
sometimes assume a package layout that does not exist here) without editing the vendored source
itself.

One file here is **generated output rather than a copied source file**, and the distinction is
stated rather than glossed: ``agentdojo_attack_corpus.json`` is what the real installed
``agentdojo`` package's own ``BaseAttack.attack()`` returned for every registered attack against
every real injection task in all four of its v1 suites. It is here because the alternative —
installing a package that pulls openai, anthropic, cohere, langchain and google-genai into this
project's interpreter so the corpus can be regenerated on every test run — is a worse trade than
vendoring the output with its generator script inside it. Generated output can go stale against its
source in a way a copied ``.py`` cannot, so ``eval/quarantine_comparison.verify_corpus_against_clone``
re-derives every template from the clone's own ``ast`` on each run and a test asserts the result.

Only vendor a file here when: (1) its licence permits redistribution, (2) the comparison is
worth the reproduction cost, and (3) the file is small and self-contained enough that vendoring
does not drag in a whole external dependency tree. Otherwise compare against the reference by
reading and citing ``file:line``, the way every other baseline in this project does.
"""

from __future__ import annotations

__all__: list[str] = []
