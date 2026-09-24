# Architecture fitness functions: enforcing layering from the import graph

**Question.** How do you *measure* architecture quality rather than argue it in a README? What does
the best available implementation check, and what does it miss?

**State of the tooling here:** none installed. `import-linter`, `pydeps`, `deptry` and `grimp` are
all absent from this machine, so anything we use has to be built on the standard library — which
suits a project whose only dependencies are pydantic and python-dateutil.

---

## The corpus's best: `minihellboy/factorminer`

`research/repos-t2/minihellboy~factorminer/scripts/check_architecture.py` — 178 lines, stdlib only,
wired into CI.

**Graph construction** by `ast`, yielding `(target_module, lineno)` per import and resolving
relative imports from the source module's own name:

```python
def iter_imports(source: str, text: str):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
```

**Three contract kinds**, all expressed as Python constants rather than a config file:

* **Layering** — `layer_for(module)` (`:81`) maps a module to a layer by prefix, and the rules are
  `if` branches: `"domain must not depend on higher layers"` (`:145`),
  `"adapter must not depend on application or interface layers"` (`:147`).
* **Independence** — `BANNED_INTERNAL_IMPORTS` (`:46`), a dict of module → replacement, checked at
  `:139`.
* **Forbidden imports** — a membership test against a frozenset plus a target, e.g.
  `"benchmark services must not depend on the runtime orchestrator"` (`:143`).

**Reporting** is every violation, sorted, with `module:line: rule: import target`, exiting non-zero.
Reporting all of them rather than failing on the first is the right call and we copied it.

**What it does not do, and these are the gaps:**

1. **No cycle detection.** It enforces a DAG by direction only. A cycle inside one layer — which is
   where cycles usually appear — is invisible to it.
2. **No distinction between a module-level and a deferred import.** Every edge counts the same.
3. **No coupling metrics.** It is a binary checker: pass or fail, no fan-in, fan-out or instability.
4. **One hop only.** It reports `A -> B`, never `A -> C -> B`, so a violation reached transitively
   is not found. This is the one that matters most, for the reason below.

---

## What ARGUS does instead

`argus/src/argus/eval/architecture.py`.

**Transitive contracts, because the live defect was two hops.** `market/*` imported `Evidence` from
`agents/analysts.py`; that module imports `argus.llm.qwen`. So:

```
>>> import argus.market.evidence     # a pure data fetcher
# ... also loaded argus.llm.qwen
```

Fetching an SEC filing loaded the model client. ARGUS's central architectural claim is that the
deterministic layers never touch the model, and a one-hop checker would have reported the
architecture clean while that was false of the very layer that gathers the facts. `Graph.reachable`
follows module-level edges to any depth; `Evidence` now lives in `truth/evidence.py`, where the
producer and the consumer both reach it without either importing the other.

**Cycles, with the distinction that decides whether they matter.** Iterative Tarjan over the whole
graph, and each cycle carries `import_time`: true only when **every** edge around it is
module-level. ARGUS has four cycles and all four are broken at load by a deferred import in one
direction — real coupling, safe ordering. Collapsing that into one number would call four safe
cycles a fault, or four faults safe.

**Coupling measured, not just contracts checked.** Martin's `I = Ce / (Ca + Ce)` per package, and
the live shape is the evidence for the criterion: `truth` has 35 dependents and depends on nothing
(I = 0.00), `cost`, `decision` and `llm` likewise; `demo` depends on seven packages and nothing
depends on it (I = 1.00). An isolated package reports `None` rather than 0.00, because zero there
would read as "maximally stable" when the figure is undefined.

**Authorisation chokepoints counted by finding the `raise`**, not by trusting a comment: an
architecture that claims a gate and logs a warning has a suggestion. Two survive the check —
`orders.py:275` and `bitget_client.py:340`.

**Where it is ahead of us.** factorminer's layering contract is a genuine ordering over named
layers; ours checks two specific properties (a model-free core, producers not importing consumers)
rather than a full layer lattice. A complete lattice would be stricter, and the reason it is not
here is that ARGUS's package graph is not a clean lattice today — `eval` legitimately depends on
almost everything because it audits it. Declaring a lattice that `eval` is exempted from would be a
contract shaped around its own violations, which is worse than not declaring one.

**Licences.** factorminer is unlicensed — study only, nothing copied; the AST-walk shape is the
obvious stdlib approach and our contracts, cycle handling and metrics are our own.
