# Vendored, verbatim, unmodified below the "VENDORED FROM HERE" marker.
#
# Source:  https://github.com/microsoft/qlib
# Path 1:  qlib/utils/__init__.py, lines 277-302 (function `parse_field`)
# Path 2:  qlib/data/data.py, lines 383-407 (class `ExpressionProvider`)
# Commit:  79633dd9506ea689e5400dea0197717b5b3d74b7 (2026-07-23) — same commit already used for
#          `qlib_expression_base.py`, `qlib_expression_ops.py` and `qlib_cs_processor.py`.
# Licence: MIT (Copyright (c) Microsoft Corporation) — full text below, per the MIT term that
#          the notice be included in every copy.
#
# These are the two real functions that turn a qlib field STRING (e.g. `"$close/Ref($close,1)"`,
# the actual syntax a qlib user writes) into a live, running Python computation:
#
#   1. `parse_field()` regex-rewrites `$close` -> `Feature("close")` and `word(` -> `Operators.word(`
#      — no allow-list, no AST, no sanitisation of any kind. Whatever survives the three regex
#      substitutions becomes a Python source string.
#   2. `ExpressionProvider.get_expression_instance()` then calls `eval()` on that string directly
#      (`expression = eval(parse_field(field))`, below), inside a bare `try/except
#      NameError/SyntaxError` — not a sandbox, just two exception types that happen to be the
#      common failure modes for a MISTYPED field, not a hostile one.
#
# `argus.eval.grammar_comparison` runs this real code, unmodified, as the decisive "does the
# baseline have a live eval/exec surface" evidence for "Typed factor grammar with no execution
# surface" in eval/standing.py: ARGUS's own `argus.research.grammar` module (walked by
# `tests/test_grammar.py`'s AST scan) contains zero calls to `eval`/`exec`/`compile`/`__import__`
# anywhere in its source, by construction — a `Window`/`Ref`/`Corr` factor expression is built from
# typed dataclasses, never from a string a caller controls. This file is qlib's own real counter-
# example, not a paraphrase of a known CWE-95 pattern: `field` is exactly the string a qlib
# strategy config supplies (field strings a user writes into a YAML handler config), and it reaches
# `eval()` unmodified inside the same class (`ExpressionProvider`) that qlib's real
# `LocalExpressionProvider` — its actual default provider, not a rarely-used code path — subclasses.
#
# The two excerpts below are NOT contiguous in the upstream source (different files, different
# line ranges); each is copied byte-for-byte from its own real location and concatenated here with
# a `# ---` divider comment (added by this vendoring, not present upstream) marking the boundary.
# `qlib_eval_surface_loader.py` in this package supplies the small set of names each snippet's own
# unmodified code references but does not define itself (`re`, `abc`, `Feature`, `PFeature`,
# `Operators`, `get_module_logger`) — never edits either body.
#
# MIT License
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ============================== VENDORED FROM HERE ==============================
# qlib/utils/__init__.py:277-302 — Copyright (c) Microsoft Corporation, MIT License.
def parse_field(field):
    # Following patterns will be matched:
    # - $close -> Feature("close")
    # - $close5 -> Feature("close5")
    # - $open+$close -> Feature("open")+Feature("close")
    # TODO: this maybe used in the feature if we want to support the computation of different frequency data
    # - $close@5min -> Feature("close", "5min")

    if not isinstance(field, str):
        field = str(field)
    # Chinese punctuation regex:
    # \u3001 -> 、
    # \uff1a -> ：
    # \uff08 -> (
    # \uff09 -> )
    chinese_punctuation_regex = r"\u3001\uff1a\uff08\uff09"
    for pattern, new in [
        (
            rf"\$\$([\w{chinese_punctuation_regex}]+)",
            r'PFeature("\1")',
        ),  # $$ must be before $
        (rf"\$([\w{chinese_punctuation_regex}]+)", r'Feature("\1")'),
        (r"(\w+\s*)\(", r"Operators.\1("),
    ]:  # Features  # Operators
        field = re.sub(pattern, new, field)
    return field


# ---
# qlib/data/data.py:383-407 -- Copyright (c) Microsoft Corporation, MIT License.
class ExpressionProvider(abc.ABC):
    """Expression provider class

    Provide Expression data.
    """

    def __init__(self):
        self.expression_instance_cache = {}

    def get_expression_instance(self, field):
        try:
            if field in self.expression_instance_cache:
                expression = self.expression_instance_cache[field]
            else:
                expression = eval(parse_field(field))
                self.expression_instance_cache[field] = expression
        except NameError as e:
            get_module_logger("data").exception(
                "ERROR: field [%s] contains invalid operator/variable [%s]" % (str(field), str(e).split()[1])
            )
            raise
        except SyntaxError:
            get_module_logger("data").exception("ERROR: field [%s] contains invalid syntax" % str(field))
            raise
        return expression
