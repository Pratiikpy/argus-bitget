"""Text in, typed tree out: a parser for the factor grammar, and a normal form for its identity.

**Why this exists.** :mod:`argus.research.grammar` had no text surface at all. Every tree in the
repository is built by Python code — the search's random generator, the hand-written factor
tables — so a model that wanted to *propose* a factor had nowhere to write it. Google's Common
Expression Language, run head to head in :mod:`argus.eval.general_grammar_comparison`, shows what
the general-purpose version of this capability looks like: untrusted text is compiled, type-checked
against declared variables and rejected with a positioned message before any data is touched, and
there is still no path from the text to the host interpreter. This module gives the factor grammar
the same shape. What was taken from CEL (``cel-expr/cel-python`` over ``google/cel-cpp``,
Apache-2.0; design only, no code): compile-then-check-then-evaluate as three separate steps,
declared-name resolution with an error that names the column, and hard input limits
(``parser/options.h``: ``expression_size_codepoint_limit`` and ``max_recursion_depth``).

**The syntax is the canonical form, nothing more.** ``Expr.canonical()`` already prints every tree
as ``name(arg,arg,...)``; :func:`parse` reads exactly that back, with whitespace allowed between
tokens so a model's pretty-printed output is accepted. ``parse(e.canonical()).canonical() ==
e.canonical()`` holds for every tree in the repository and is tested. There is no second syntax to
drift from the first.

**No execution surface, and the same test proves it.** The parser is a hand-written tokenizer and
recursive descent over a fixed table of names. It never evaluates, compiles, imports or reflects;
``tests/test_grammar_text.py`` runs the same AST scan over this file that
``tests/test_grammar.py`` runs over the grammar. A name not in the table is an error, not a lookup.

**Normal form: the part SymPy does better.** ``canonical()`` sorts commutative arguments, so
``add(a,b)`` and ``add(b,a)`` are one trial. It does not know that ``neg(neg(x))`` is ``x``, that
``sub(a,neg(b))`` is ``add(a,b)``, that ``lt(b,a)`` is ``gt(a,b)``, or that ``add(add(a,b),c)`` is
``add(a,add(b,c))``; SymPy's automatic canonicalisation does, and on the comparison's duplicate
corpus it merged pairs ARGUS paid for twice. :func:`normal_form` adds those four rewrites. The first
three are exact in IEEE-754 (negation and operand swap of a comparison introduce no rounding).
Re-association is not bit-exact — ``(a+b)+c`` and ``a+(b+c)`` can differ in the last place — so the
normal form is an identity of *hypotheses*, the same convention ``Const.canonical`` already applies
when it rounds to six significant digits. It is offered for trial accounting and is not wired into
``canonical()``, which existing artefacts key on.
"""

from __future__ import annotations

from dataclasses import dataclass

from argus.research.grammar import (
    MAX_DEPTH,
    BinOp,
    Const,
    Corr,
    CrossRank,
    CrossScale,
    Delay,
    Expr,
    Field,
    GrammarError,
    HasDiscovery,
    IfElse,
    InPhase,
    Ref,
    Return,
    Signal,
    UnOp,
    Window,
    validate,
)
from argus.truth.clocks import SessionPhase

MAX_TEXT = 4096
"""Longest input accepted, in characters. ``MAX_SIZE`` is 96 nodes, and the longest canonical
spelling of a 96-node tree in this grammar is well under this. CEL's own default limit is far larger
(100,000 codepoints) because its expressions are general; ours are not."""

MAX_NESTING = MAX_DEPTH
"""Parenthesis nesting at which parsing stops. :func:`validate` refuses anything deeper anyway;
stopping in the parser means a pathological input cannot reach Python's own recursion limit."""


class GrammarTextError(GrammarError):
    """The text is not a well-formed, well-typed factor. Carries the column it went wrong at."""

    def __init__(self, message: str, text: str, position: int) -> None:
        self.position = position
        self.text = text
        caret = " " * max(0, position) + "^"
        excerpt = text if len(text) <= 120 else text[:120] + "..."
        super().__init__(f"{message} at column {position + 1}\n  {excerpt}\n  {caret}")


# --- tokenizer --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str       # "name" | "number" | "(" | ")" | "," | "end"
    text: str
    position: int


_NAME_START = frozenset("abcdefghijklmnopqrstuvwxyz_")
_NAME_REST = _NAME_START | frozenset("0123456789")
_DIGITS = frozenset("0123456789")


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch in "(),":
            tokens.append(_Token(ch, ch, i))
            i += 1
            continue
        if ch in _NAME_START:
            j = i + 1
            while j < n and text[j] in _NAME_REST:
                j += 1
            tokens.append(_Token("name", text[i:j], i))
            i = j
            continue
        if ch in _DIGITS or ch in "-+.":
            j = i + 1
            while j < n and (text[j] in _DIGITS or text[j] in ".eE"
                             or (text[j] in "+-" and text[j - 1] in "eE")):
                j += 1
            tokens.append(_Token("number", text[i:j], i))
            i = j
            continue
        raise GrammarTextError(f"unexpected character {ch!r}", text, i)
    tokens.append(_Token("end", "", n))
    return tokens


# --- parser -----------------------------------------------------------------------------------

_FIELDS = {f.value: f for f in Field}
_PHASES = {p.value: p for p in SessionPhase}
_BINARY = frozenset(BinOp.ARITH + BinOp.COMPARE + BinOp.LOGIC)
_UNARY = frozenset(UnOp.OPS)
_WINDOW = frozenset(Window.OPS)


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.k = 0

    def _peek(self) -> _Token:
        return self.tokens[self.k]

    def _take(self, kind: str) -> _Token:
        tok = self.tokens[self.k]
        if tok.kind != kind:
            want = "end of input" if kind == "end" else repr(kind)
            got = "end of input" if tok.kind == "end" else repr(tok.text)
            raise GrammarTextError(f"expected {want}, found {got}", self.text, tok.position)
        self.k += 1
        return tok

    def _fail(self, message: str, tok: _Token) -> GrammarTextError:
        return GrammarTextError(message, self.text, tok.position)

    def parse(self) -> Expr:
        expr = self._expr(0)
        self._take("end")
        return expr

    def _expr(self, depth: int) -> Expr:
        tok = self._peek()
        if depth > MAX_NESTING:
            raise self._fail(f"nesting deeper than {MAX_NESTING}", tok)
        if tok.kind == "number":
            self.k += 1
            return Const(self._finite(tok))
        if tok.kind != "name":
            got = "end of input" if tok.kind == "end" else repr(tok.text)
            raise self._fail(f"expected an expression, found {got}", tok)
        self.k += 1
        if self._peek().kind != "(":
            field = _FIELDS.get(tok.text)
            if field is None:
                raise self._fail(
                    f"unknown field {tok.text!r}; declared fields: {sorted(_FIELDS)}", tok
                )
            return Ref(field)
        self._take("(")
        node = self._call(tok, depth)
        self._take(")")
        return node

    def _args(self, count: int, name: _Token, depth: int,
              lookback_first: bool = False) -> tuple[int | None, list[Expr]]:
        """Read ``count`` arguments; the first is an integer lookback when ``lookback_first``."""
        lookback: int | None = None
        exprs: list[Expr] = []
        for index in range(count):
            if index:
                self._take(",")
            if index == 0 and lookback_first:
                lookback = self._integer(self._take("number"))
            else:
                exprs.append(self._expr(depth + 1))
        if self._peek().kind == ",":
            raise self._fail(f"{name.text} takes {count} argument(s)", self._peek())
        return lookback, exprs

    def _call(self, name: _Token, depth: int) -> Expr:
        try:
            return self._dispatch(name, depth)
        except GrammarTextError:
            raise
        except GrammarError as exc:
            # The node's own type check, re-raised at the column the node starts: CEL's
            # "found no matching overload ... <input>:1:7" rather than a bare message.
            raise self._fail(str(exc), name) from exc

    def _dispatch(self, name: _Token, depth: int) -> Expr:
        fn = name.text
        if fn == "signal":
            _, (x,) = self._args(1, name, depth)
            return Signal(x)
        if fn == "if":
            _, (c, t, o) = self._args(3, name, depth)
            return IfElse(c, t, o)
        if fn in _BINARY:
            _, (a, b) = self._args(2, name, depth)
            return BinOp(fn, a, b)
        if fn in _UNARY:
            _, (x,) = self._args(1, name, depth)
            return UnOp(fn, x)
        if fn in _WINDOW:
            n, (x,) = self._args(2, name, depth, lookback_first=True)
            return Window(fn, _known(n), x)
        if fn == "delay":
            n, (x,) = self._args(2, name, depth, lookback_first=True)
            return Delay(_known(n), x)
        if fn == "corr":
            n, (a, b) = self._args(3, name, depth, lookback_first=True)
            return Corr(_known(n), a, b)
        if fn == "ret":
            return Return(self._integer(self._take("number")))
        if fn == "crossrank":
            _, (x,) = self._args(1, name, depth)
            return CrossRank(x)
        if fn == "crossscale":
            _, (x,) = self._args(1, name, depth)
            return CrossScale(x)
        if fn == "in_phase":
            tok = self._take("name")
            phase = _PHASES.get(tok.text)
            if phase is None:
                raise self._fail(
                    f"unknown session phase {tok.text!r}; declared: {sorted(_PHASES)}", tok
                )
            return InPhase(phase)
        if fn == "has_discovery":
            return HasDiscovery()
        raise self._fail(f"unknown function {fn!r}", name)

    def _finite(self, tok: _Token) -> float:
        try:
            value = float(tok.text)
        except ValueError:
            raise self._fail(f"malformed number {tok.text!r}", tok) from None
        if value != value or value in (float("inf"), float("-inf")):
            raise self._fail("constants must be finite", tok)
        return value

    def _integer(self, tok: _Token) -> int:
        if not tok.text.isdigit():
            raise self._fail(f"a lookback must be a whole number of bars, got {tok.text!r}", tok)
        return int(tok.text)


def _known(n: int | None) -> int:
    if n is None:  # pragma: no cover - _args always reads the lookback when asked to
        raise GrammarError("lookback missing")
    return n


def parse(text: str) -> Expr:
    """Parse the canonical spelling of an expression. Raises :class:`GrammarTextError`.

    Every construction check the grammar's own nodes run (kinds, lookback bounds, operator names)
    runs here too, because parsing builds real nodes; the error is re-raised at the column the
    offending node starts.
    """
    if len(text) > MAX_TEXT:
        raise GrammarTextError(f"input is {len(text)} characters; the limit is {MAX_TEXT}",
                               text[:MAX_TEXT], MAX_TEXT)
    return _Parser(text).parse()


def parse_factor(text: str, **limits: int) -> Signal:
    """Parse and validate a complete factor: rooted in ``signal``, within depth, size and cost."""
    return validate(parse(text), **limits)


# --- normal form ------------------------------------------------------------------------------

_ASSOCIATIVE = frozenset({"add", "mul", "and", "or"})


def normal_form(expr: Expr) -> str:
    """A spelling-independent identity stronger than ``canonical()``. See the module docstring.

    Rewrites applied, bottom-up: ``neg(neg(x)) -> x``; ``sub(a, neg(b)) -> add(a, b)``;
    ``lt(a, b) -> gt(b, a)``; associative ``add/mul/and/or`` chains flattened and their operands
    sorted. Every other node keeps ``canonical()``'s spelling, including its constant rounding.
    """
    return _nf(expr)


def _nf(expr: Expr) -> str:
    if isinstance(expr, UnOp) and expr.op == "neg":
        inner = expr.operand
        if isinstance(inner, UnOp) and inner.op == "neg":
            return _nf(inner.operand)
        return f"neg({_nf(inner)})"
    if isinstance(expr, BinOp):
        if expr.op == "sub" and isinstance(expr.right, UnOp) and expr.right.op == "neg":
            return _nf(BinOp("add", expr.left, expr.right.operand))
        if expr.op == "lt":
            return f"gt({_nf(expr.right)},{_nf(expr.left)})"
        if expr.op in _ASSOCIATIVE:
            terms = sorted(_flatten(expr.op, expr))
            return f"{expr.op}({','.join(terms)})"
        return f"{expr.op}({_nf(expr.left)},{_nf(expr.right)})"
    if isinstance(expr, Window):
        return f"{expr.op}({expr.lookback},{_nf(expr.operand)})"
    if isinstance(expr, Delay):
        return f"delay({expr.lookback},{_nf(expr.operand)})"
    if isinstance(expr, Corr):
        legs = sorted((_nf(expr.left), _nf(expr.right)))
        return f"corr({expr.lookback},{legs[0]},{legs[1]})"
    if isinstance(expr, IfElse):
        return f"if({_nf(expr.condition)},{_nf(expr.then)},{_nf(expr.otherwise)})"
    if isinstance(expr, UnOp):
        return f"{expr.op}({_nf(expr.operand)})"
    if isinstance(expr, Signal):
        return f"signal({_nf(expr.operand)})"
    if isinstance(expr, CrossRank):
        return f"crossrank({_nf(expr.operand)})"
    if isinstance(expr, CrossScale):
        return f"crossscale({_nf(expr.operand)})"
    return expr.canonical()


def _flatten(op: str, expr: Expr) -> list[str]:
    if isinstance(expr, BinOp) and expr.op == op:
        return _flatten(op, expr.left) + _flatten(op, expr.right)
    if op == "add" and isinstance(expr, BinOp) and expr.op == "sub" \
            and isinstance(expr.right, UnOp) and expr.right.op == "neg":
        return _flatten(op, BinOp("add", expr.left, expr.right.operand))
    return [_nf(expr)]


__all__ = [
    "MAX_NESTING",
    "MAX_TEXT",
    "GrammarTextError",
    "normal_form",
    "parse",
    "parse_factor",
]
