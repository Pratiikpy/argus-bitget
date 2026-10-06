"""Numbers as a trader reads them.

Python's ``g`` format switches to scientific notation once a number has more integer digits than
the requested precision, so ``f"{86473.2:,.4g}"`` prints ``8.647e+04``: a daily candle read
"opened 8.473e+04, closed 8.647e+04" (round 44 hostile, minor 11). These keep the significant
figures and never use an exponent.
"""

from __future__ import annotations

import math

__all__ = ["price", "sig"]


def sig(x: float, n: int = 4) -> str:
    """``x`` to ``n`` significant figures, with thousands separators and no exponent: 86,470,
    0.001235, 22.9. Whole digits are never dropped, so 86,473.2 to four figures is 86,473."""
    if not math.isfinite(x):
        return str(x)
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = max(0, n - 1 - magnitude)
    said = f"{x:,.{decimals}f}"
    # as ``g`` does, no trailing zeros: 0.06 gwei, not 0.0600
    return said.rstrip("0").rstrip(".") if "." in said else said


def price(x: float) -> str:
    """A price: two decimals from one unit up, four significant figures below it."""
    if not math.isfinite(x):
        return str(x)
    return f"{x:,.2f}" if abs(x) >= 1 else sig(x, 4)
