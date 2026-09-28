"""The console's modules keep to their concern (audit finding 169).

`lui/` holds four kinds of module — reading a question, answering it, reaching the trader, and
rendering pages — side by side in one package. They are kept apart by these rules rather than by
folders, because the modules are imported by name across the project, the deployment and the
tests, and a rule enforced on every run protects the separation better than a move would:

* a channel (the web console, the CLI, the MCP server, the Telegram bots) is imported by no other
  console module: nothing answers differently because of how the question arrived;
* a page renderer is imported only by a channel or another page: an answer never builds HTML;
* a reader imports only other readers and the research package's parser: reading a question never
  depends on an engine's answer.

Two breaches existed when the rules were written, and were fixed rather than excused: the answer
engine imported `/wrong`'s page to collect the findings (the collection is now
`lui/corrections.py`), and the research task drew its own SVG charts (now `lui/task_page.py`).
"""

from __future__ import annotations

import ast
from pathlib import Path

LUI = Path(__file__).resolve().parents[1] / "src" / "argus" / "lui"
CHANNELS = {"server", "cli", "mcp_server", "telegram_bot", "selfhost", "__main__"}
PAGES = {"design", "task_page", "status_page", "proof_page", "corrections_page", "agent_page",
         "brand_page", "materials_page", "architecture_page", "factors_page",
         "policy_page"}
READERS = {"question", "normalise", "phrasebook", "ngram", "kindmodel", "router", "arbiter"}


def _imports(path: Path) -> set[str]:
    """The `argus.lui` modules a module imports (its package's own `research` counts as one)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("argus.lui"):
            parts = node.module.split(".")
            if len(parts) >= 3:
                found.add(parts[2])
            else:
                found.update(alias.name for alias in node.names)
    return found


def _graph() -> dict[str, set[str]]:
    return {p.stem: _imports(p) - {p.stem} for p in LUI.glob("*.py")}


def test_no_console_module_imports_a_channel() -> None:
    graph = _graph()
    assert {m: sorted(d & CHANNELS) for m, d in graph.items()
            if m not in CHANNELS and d & CHANNELS} == {}


def test_only_channels_and_pages_import_a_page() -> None:
    graph = _graph()
    assert {m: sorted(d & PAGES) for m, d in graph.items()
            if m not in CHANNELS | PAGES and d & PAGES} == {}


def test_readers_depend_only_on_readers_and_the_parser() -> None:
    graph = _graph()
    allowed = READERS | {"research"}
    assert {m: sorted(d - allowed) for m, d in graph.items() if m in READERS and d - allowed} == {}


def test_every_module_is_in_a_concern() -> None:
    engines = {p.stem for p in LUI.glob("*.py")} - CHANNELS - PAGES - READERS - {"__init__"}
    assert "answer" in engines and "task" in engines and "research" not in engines
