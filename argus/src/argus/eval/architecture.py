"""Agent architecture quality, measured from the import graph rather than asserted in prose.

Track 2 names **four** things its judges score: *"Paper trading Sharpe, max drawdown, win rate;
decision explainability; **Agent architecture quality**; risk control layer effectiveness."*
`eval/themeaudit.py` probed three of them. The fourth had no probe, and the coverage test written
to guard that quoted all four in its docstring and then asserted a subset of three — so the gap
passed its own check.

Architecture quality is usually argued in a README. These are the parts of it that are **facts about
the import graph**, and a fact can be checked:

1. **Import-time cycles.** A cycle where every edge is module-level can break at import and forces
   an import order nobody wrote down. A cycle broken in one direction by a deferred, in-function
   import is a different thing: the coupling is real, the ordering is safe, and it is usually
   deliberate. Reporting them as one number would call four safe cycles a fault or four faults
   safe.
2. **The deterministic core.** ARGUS's central claim is that the model interprets numbers and never
   produces them. That is checkable: no module under `truth`, `cost`, `risk`, `decision` or
   `backtest` may import `argus.llm` or `argus.agents`, directly or transitively.
3. **Producers must not import consumers.** `market` gathers facts; `agents` reads them. A
   dependency the other way inverts the layering.
4. **Authorisation chokepoints.** The number of places that *raise* rather than warn when an order
   arrives without the hash of the verdict that approved it.
5. **Instability**, in Robert Martin's sense: ``I = Ce / (Ca + Ce)``. A package everything depends
   on and which depends on nothing is stable and should be; one in the middle with high instability
   and high fan-in is where change hurts.

**This found a real defect the first time it ran, which is why it exists.** ``market/*`` imported
``Evidence`` from ``agents/analysts.py``, and that module imports ``argus.llm.qwen`` — so

    >>> import argus.market.evidence      # a pure data fetcher
    # ... also loaded argus.llm.qwen

Fetching an SEC filing loaded the model client. The claim in (2) was verified for five packages and
quietly false for the one that *gathers the facts the claim is about*. `Evidence` now lives in
`truth/evidence.py`, where the producer and the consumer can both reach it without either importing
the other.

**Prior art** (`research/architecture/architecture-fitness.md`): no enforcement tool is installed
here, and the corpus's best implementation is `factorminer`'s
`scripts/check_architecture.py` — 178 lines, stdlib `ast`, layering plus forbidden-import plus
independence contracts, every violation reported rather than the first. It has **no cycle detection
and no coupling metrics**, which is most of what this module adds.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parents[3] / "data"
REPORT_PATH = DATA / "architecture.json"

DETERMINISTIC = ("truth", "cost", "risk", "decision", "backtest")
"""Packages that must compute rather than ask. The deterministic core.

`execution` is deliberately absent: it holds the venue client, which is I/O and not arithmetic.
`market` is absent for the same reason — it fetches. What matters is that neither reaches the model
for a *number*, and (2) checks the stronger, simpler property on the packages where it should hold
absolutely.
"""

MODEL_FACING = ("argus.llm", "argus.agents")
"""What the deterministic core may not import at any depth."""

PRODUCER_BEFORE_CONSUMER = (("market", "argus.agents"),)
"""(package, forbidden prefix) — a producer importing its consumer has inverted the layering."""


class ArchitectureError(RuntimeError):
    """Raised rather than reporting a clean architecture from a graph that failed to build."""


@dataclass(frozen=True, slots=True)
class Edge:
    """One import, and whether it happens when the module loads."""

    source: str
    target: str
    line: int
    module_level: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "from": self.source, "to": self.target, "line": self.line,
            "module_level": self.module_level,
        }


@dataclass(frozen=True, slots=True)
class Cycle:
    """A set of modules that depend on each other, and whether it can break at import time."""

    members: tuple[str, ...]
    import_time: bool
    """True when **every** edge around the cycle is module-level.

    The distinction the corpus's best checker does not make. A cycle held open by one deferred
    import cannot fail at load; one that is module-level all the way round is an ordering
    constraint nobody declared, and it breaks the first time someone imports the wrong member first.
    """

    def as_dict(self) -> dict[str, Any]:
        return {"members": list(self.members), "import_time": self.import_time}


@dataclass(frozen=True, slots=True)
class Violation:
    """One contract, broken, with the import that broke it."""

    contract: str
    detail: str
    where: str

    def as_dict(self) -> dict[str, Any]:
        return {"contract": self.contract, "detail": self.detail, "where": self.where}


@dataclass(frozen=True, slots=True)
class Graph:
    modules: tuple[str, ...]
    edges: tuple[Edge, ...]

    def out_of(self, module: str) -> set[str]:
        return {e.target for e in self.edges if e.source == module}

    @property
    def packages(self) -> tuple[str, ...]:
        return tuple(sorted({m.split(".")[1] for m in self.modules if m.count(".") >= 1}))

    def reachable(self, start: str) -> set[str]:
        """Every module reachable from ``start``, following module-level edges only.

        Module-level, because that is what "importing this loads that" means. A deferred import is
        a runtime dependency and does not put the target into `sys.modules` at load.
        """
        seen: set[str] = set()
        stack = [start]
        by_source: dict[str, list[str]] = {}
        for edge in self.edges:
            if edge.module_level:
                by_source.setdefault(edge.source, []).append(edge.target)
        while stack:
            node = stack.pop()
            for target in by_source.get(node, ()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return seen


def _module_name(path: Path, root: Path) -> str:
    """The dotted name of ``path`` relative to the package ``root``.

    ``root``, not the module-level ``SOURCE``: `build` takes a root so it can be pointed at a
    synthetic tree, and a namer that ignored it made that parameter a lie — every test that tried
    raised ``ValueError: not in the subpath``. Caught by the first test written against it.
    """
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute(node: ast.ImportFrom, importer: str) -> str | None:
    """The absolute module a `from ... import ...` refers to, relative ones included.

    **A blind spot this closes.** The first version accepted only ``node.level == 0``, so every
    relative import (`from .clocks import DualClock`) was dropped from the graph — invisible to the
    cycle detector, to the deterministic-core rule and to the coupling metrics alike. It was latent
    rather than live: this codebase uses absolute imports throughout, so the graph was complete by
    convention rather than by construction. A checker that is correct only while a convention holds
    is a checker that stops being correct without telling anyone.

    ``level`` counts the leading dots. One dot means the importer's own package, two means its
    parent, and so on, so the base is the importer's name with ``level`` trailing segments removed.
    """
    if node.level == 0:
        return node.module or None
    parts = importer.split(".")
    if node.level > len(parts):
        return None
    base = parts[: len(parts) - node.level]
    if node.module:
        base = [*base, node.module]
    return ".".join(base) or None


def build(root: Path = SOURCE) -> Graph:
    """Parse every module and record its imports, noting which happen at load time."""
    files = [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]
    if not files:
        raise ArchitectureError(f"no python files under {root}; the graph would be empty")
    names = {_module_name(p, root): p for p in files}
    edges: list[Edge] = []
    for name, path in names.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                resolved = _absolute(node, name)
                if resolved is None:
                    continue
                targets = [resolved]
            else:
                continue
            for raw in targets:
                if not raw.startswith("argus"):
                    continue
                # Resolve `argus.a.b.C` to the deepest real module, so an imported symbol does not
                # look like a module that does not exist.
                target = raw
                while target and target not in names:
                    target = target.rsplit(".", 1)[0] if "." in target else ""
                if target and target != name:
                    edges.append(Edge(name, target, node.lineno, id(node) in top))
    return Graph(modules=tuple(names), edges=tuple(edges))


def cycles(graph: Graph) -> tuple[Cycle, ...]:
    """Every strongly connected component larger than one module (Tarjan, iterative).

    Iterative rather than recursive: the graph is 145 modules today and a recursive Tarjan on a
    deep chain would need the interpreter's recursion limit raised, which is a landmine for a check
    that is supposed to be boring.
    """
    adjacency: dict[str, list[str]] = {m: [] for m in graph.modules}
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)

    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = 0
    found: list[list[str]] = []

    for root in graph.modules:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child = work[-1]
            if child == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)
            if child < len(adjacency[node]):
                work[-1] = (node, child + 1)
                nxt = adjacency[node][child]
                if nxt not in index:
                    work.append((nxt, 0))
                elif nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                if low[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1:
                        found.append(component)
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])

    out: list[Cycle] = []
    for component in found:
        members = frozenset(component)
        inside = [e for e in graph.edges if e.source in members and e.target in members]
        out.append(Cycle(
            members=tuple(sorted(component)),
            import_time=all(e.module_level for e in inside) if inside else False,
        ))
    return tuple(sorted(out, key=lambda c: (not c.import_time, c.members)))


def contract_violations(graph: Graph) -> tuple[Violation, ...]:
    """Check every declared contract and report all breaches, never only the first."""
    out: list[Violation] = []
    for module in graph.modules:
        parts = module.split(".")
        package = parts[1] if len(parts) > 1 else ""
        if package in DETERMINISTIC:
            for target in sorted(graph.reachable(module)):
                if target.startswith(MODEL_FACING):
                    out.append(Violation(
                        contract="deterministic core is model-free",
                        detail=f"{module} reaches {target}",
                        where=module,
                    ))
                    break
        for producer, forbidden in PRODUCER_BEFORE_CONSUMER:
            if package == producer:
                for edge in graph.edges:
                    if edge.source == module and edge.target.startswith(forbidden):
                        out.append(Violation(
                            contract="a producer must not import its consumer",
                            detail=f"{module} imports {edge.target}",
                            where=f"{module}:{edge.line}",
                        ))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class PackageMetrics:
    """Martin's coupling figures for one package."""

    name: str
    afferent: int
    """How many modules outside this package depend on it. Ca."""

    efferent: int
    """How many packages this one depends on. Ce."""

    modules: int

    @property
    def instability(self) -> float | None:
        total = self.afferent + self.efferent
        return None if total == 0 else self.efferent / total

    def as_dict(self) -> dict[str, Any]:
        return {
            "package": self.name, "modules": self.modules,
            "afferent": self.afferent, "efferent": self.efferent,
            "instability": None if self.instability is None else round(self.instability, 3),
        }


def metrics(graph: Graph) -> tuple[PackageMetrics, ...]:
    out: list[PackageMetrics] = []
    for package in graph.packages:
        prefix = f"argus.{package}"
        inside = [m for m in graph.modules if m == prefix or m.startswith(prefix + ".")]
        efferent = {
            e.target.split(".")[1] for e in graph.edges
            if e.source in inside and not e.target.startswith(prefix) and e.target.count(".") >= 1
        }
        afferent = {
            e.source for e in graph.edges
            if e.target in inside and not e.source.startswith(prefix)
        }
        out.append(PackageMetrics(
            name=package, afferent=len(afferent), efferent=len(efferent), modules=len(inside),
        ))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Report:
    graph: Graph
    cycles: tuple[Cycle, ...]
    violations: tuple[Violation, ...]
    packages: tuple[PackageMetrics, ...] = field(default_factory=tuple)

    @property
    def import_time_cycles(self) -> tuple[Cycle, ...]:
        return tuple(c for c in self.cycles if c.import_time)

    @property
    def sound(self) -> bool:
        """No broken contract and no cycle that can break at import."""
        return not self.violations and not self.import_time_cycles

    @property
    def verdict(self) -> str:
        parts = [
            f"{len(self.graph.modules)} modules, {len(self.graph.edges)} internal imports"
        ]
        if self.violations:
            names = "; ".join(f"{v.contract} — {v.detail}" for v in self.violations[:4])
            parts.append(f"{len(self.violations)} contract violation(s): {names}")
        else:
            parts.append(
                "every declared contract holds: the deterministic core reaches no model-facing "
                "module at any depth, and no producer imports its consumer"
            )
        if self.import_time_cycles:
            parts.append(
                f"{len(self.import_time_cycles)} cycle(s) are module-level the whole way round and "
                f"can break at import"
            )
        elif self.cycles:
            parts.append(
                f"{len(self.cycles)} dependency cycle(s), each broken at load by a deferred import "
                f"in one direction — real coupling, safe ordering, and reported rather than scored "
                f"as clean"
            )
        else:
            parts.append("no dependency cycles")
        return ". ".join(parts) + "."

    def render(self) -> str:
        lines = ["ARCHITECTURE", "", f"  {self.verdict}", ""]
        for violation in self.violations:
            lines.append(f"  VIOLATION  {violation.contract}")
            lines.append(f"             {violation.detail} ({violation.where})")
        for cycle in self.cycles:
            kind = "IMPORT-TIME" if cycle.import_time else "deferred"
            lines.append(f"  CYCLE {kind:<12} {' <-> '.join(cycle.members)}")
        lines += ["", f"{'package':>12}{'modules':>9}{'Ca':>6}{'Ce':>5}{'instability':>13}"]
        for package in sorted(self.packages, key=lambda p: -(p.instability or 0)):
            shown = "n/a" if package.instability is None else f"{package.instability:.2f}"
            lines.append(
                f"{package.name:>12}{package.modules:>9}{package.afferent:>6}"
                f"{package.efferent:>5}{shown:>13}"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "modules": len(self.graph.modules),
            "internal_imports": len(self.graph.edges),
            "sound": self.sound,
            "violations": [v.as_dict() for v in self.violations],
            "cycles": [c.as_dict() for c in self.cycles],
            "import_time_cycles": len(self.import_time_cycles),
            "packages": [p.as_dict() for p in self.packages],
            "verdict": self.verdict,
        }


def audit(root: Path = SOURCE) -> Report:
    graph = build(root)
    return Report(
        graph=graph, cycles=cycles(graph), violations=contract_violations(graph),
        packages=metrics(graph),
    )


def authorisation_chokepoints(paths: Sequence[Path] | None = None) -> tuple[str, ...]:
    """Where an order without its approving verdict is *refused*, not merely logged.

    Counted by finding the raises rather than by trusting a comment: an architecture that claims a
    chokepoint and warns instead of raising has a suggestion, not a gate.
    """
    targets = paths or (
        SOURCE / "execution" / "orders.py",
        SOURCE / "execution" / "bitget_client.py",
    )
    found: list[str] = []
    for path in targets:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            segment = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
            if "approved_intent_hash" in segment or "unauthorised" in segment.lower():
                found.append(f"{path.name}:{node.lineno}")
    return tuple(found)


def main() -> int:  # pragma: no cover - CLI
    report = audit()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    print(report.render())
    print("\nwritten to " + str(REPORT_PATH))
    return 0 if report.sound else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "DETERMINISTIC",
    "MODEL_FACING",
    "ArchitectureError",
    "Cycle",
    "Edge",
    "Graph",
    "PackageMetrics",
    "Report",
    "Violation",
    "audit",
    "authorisation_chokepoints",
    "build",
    "contract_violations",
    "cycles",
    "metrics",
]
