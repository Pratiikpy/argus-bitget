"""A deploy stops before it ships a module the host cannot import (2026-10-06: numpy, HTTP 500)."""

from __future__ import annotations

from pathlib import Path

from argus.demo.deploysync import undeclared_imports


def test_a_served_module_needing_an_undeclared_package_is_named(tmp_path: Path) -> None:
    pkg = tmp_path / "argus"
    (pkg / "lui").mkdir(parents=True)
    (pkg / "eval").mkdir()
    (pkg / "lui" / "a.py").write_text(
        '"""from the docstring is prose."""\nimport json\nimport numpy as np\n'
        "from argus.lui import b\n\n\ndef f() -> None:\n    import scipy\n", encoding="utf-8")
    (pkg / "eval" / "c.py").write_text("import pandas\n", encoding="utf-8")
    req = tmp_path / "requirements.txt"
    req.write_text("cryptography>=43\n", encoding="utf-8")
    assert undeclared_imports(pkg, req) == ["lui/a.py: numpy"]
    req.write_text("cryptography>=43\nnumpy>=2.0\n", encoding="utf-8")
    assert undeclared_imports(pkg, req) == []
