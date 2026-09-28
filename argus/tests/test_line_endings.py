"""Every file ARGUS writes as text is written with ``\\n`` line endings on every system.

Python's text mode translates ``\\n`` to the platform's separator, so a report written on Windows
was CRLF and the same report written on Linux was LF. CI regenerates the register, the anchor
check and the documentation claims and then requires that nothing committed changed; run 2026-09-28
(36360268957) failed on Ubuntu with every line of those four files rewritten and not one figure
different. A write without an explicit ``newline`` is the whole of that defect, so it is refused
here rather than found again by CI."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "argus"
SKIPPED = ("vendor", "baselines")
"""Vendored excerpts and reimplementations pinned by the hash of their exact bytes."""
NOT_FILES = ("gzip", "os", "io", "tarfile", "zipfile", "webbrowser", "lzma", "bz2")


def _text_mode_write(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "write_text":
        return True
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(
        func, ast.Attribute) else ""
    if name != "open":
        return False
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
            and func.value.id in NOT_FILES:
        return False
    mode: ast.expr | None = None
    if isinstance(func, ast.Name) and len(call.args) >= 2:
        mode = call.args[1]
    elif isinstance(func, ast.Attribute) and call.args:
        mode = call.args[0]
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
        return False
    return any(c in mode.value for c in "wax") and "b" not in mode.value


def test_every_text_write_names_its_line_ending() -> None:
    unset = []
    for path in sorted(SRC.rglob("*.py")):
        if any(part in SKIPPED for part in path.relative_to(SRC).parts):
            continue
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if isinstance(node, ast.Call) and _text_mode_write(node) and not any(
                    k.arg in ("newline", None) for k in node.keywords):
                unset.append(f"{path.relative_to(SRC.parent).as_posix()}:{node.lineno}")
    assert not unset, unset


def test_the_check_would_catch_a_bare_write() -> None:
    bare = ast.parse('Path("x").write_text("a", encoding="utf-8")').body[0]
    fixed = ast.parse('Path("x").write_text("a", encoding="utf-8", newline="\\n")').body[0]
    appended = ast.parse('open(p, "a", encoding="utf-8")').body[0]
    binary = ast.parse('open(p, "wb")').body[0]
    calls = [n.value for n in (bare, fixed, appended, binary) if isinstance(n, ast.Expr)]
    assert all(isinstance(c, ast.Call) for c in calls)
    flagged = [_text_mode_write(c) and not any(k.arg == "newline" for k in c.keywords)
               for c in calls if isinstance(c, ast.Call)]
    assert flagged == [True, False, True, False]


def test_no_source_file_carries_a_control_character() -> None:
    """A regex's word boundary written through a shell heredoc arrived as a literal backspace
    (0x08) in lui/thesis.py on 2026-09-28: the pattern still compiled and silently matched less.
    Nothing but tab, newline and carriage return belongs in source."""
    allowed = {9, 10, 13}
    found = []
    for path in sorted(SRC.rglob("*.py")):
        data = path.read_bytes()
        bad = sorted({b for b in data if b < 32 and b not in allowed})
        if bad:
            found.append(f"{path.relative_to(SRC.parent).as_posix()}: {bad}")
    assert not found, found
