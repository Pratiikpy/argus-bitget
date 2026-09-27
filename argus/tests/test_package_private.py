"""A leading underscore means package-private, and product code outside the package does not reach
for it (audit finding 168).

On 2026-09-27 there were 55 such uses across 35 names — `lui/kindmodel.py` reading
`research._pairs`, `research._SHOCK_WEIGHT` and `research._detect`, the console's pages reading
`research._clean` — so a rename inside one module could break another with nothing marking the
dependency. Each was made public under a name that says what it is. The evaluation harnesses
(`eval/`) may still reach in: measuring a private step is what they are for.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ROOT = SRC / "argus"


def _module(path: Path, src: Path) -> str:
    parts = list(path.relative_to(src).with_suffix("").parts)
    return ".".join(parts)


def _is_module(dotted: str, src: Path) -> bool:
    base = src / Path(*dotted.split("."))
    return base.is_dir() or base.with_suffix(".py").exists()


def _package(dotted: str, src: Path) -> str:
    return dotted if (src / Path(*dotted.split("."))).is_dir() else dotted.rsplit(".", 1)[0]


def cross_package_private_uses(root: Path = ROOT) -> list[str]:
    """``module: owner.name`` for every private name used from outside the owner's package."""
    src = root.parent
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if {"vendor", "baselines", "eval"} & set(path.relative_to(root).parts):
            continue
        me = _module(path, src)
        mine = me.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("argus"):
                for alias in node.names:
                    target = f"{node.module}.{alias.name}"
                    if _is_module(target, src):
                        modules[alias.asname or alias.name] = target
                    elif alias.name.startswith("_") and not alias.name.startswith("__") \
                            and _package(node.module, src) != mine:
                        found.append(f"{me}: {node.module}.{alias.name}")
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr.startswith("_")
                    and not node.attr.startswith("__")):
                continue
            chain: list[str] = []
            cur: ast.AST = node.value
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in modules:
                owner = ".".join([modules[cur.id], *reversed(chain)])
                if _is_module(owner, src) and _package(owner, src) != mine:
                    found.append(f"{me}: {owner}.{node.attr}")
    return found


def test_no_product_module_uses_another_packages_private_names() -> None:
    assert cross_package_private_uses() == []


def test_the_check_sees_both_spellings(tmp_path: Path) -> None:
    pkg = tmp_path / "argus"
    (pkg / "a").mkdir(parents=True)
    (pkg / "b").mkdir()
    for d in (pkg, pkg / "a", pkg / "b"):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a" / "m.py").write_text("_x = 1\n", encoding="utf-8")
    (pkg / "b" / "n.py").write_text(
        "from argus.a.m import _x\nfrom argus.a import m\ny = m._x\n", encoding="utf-8")
    assert cross_package_private_uses(pkg) == ["argus.b.n: argus.a.m._x",
                                               "argus.b.n: argus.a.m._x"]
