"""Public functions: one per step size strategy."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence, Union

import numpy as np
import sympy as sp

from . import _core
from ._core import ComputationError
from ._expr import ExpressionError, parse_function, parse_number
from .result import Solution

Expression = Union[str, sp.Basic, Real]
Number = Union[Real, str]

DEFAULT_MAX_STEPS = 100_000

_T = _core.TIME_SYMBOL
_X = _core.STATE_SYMBOLS[0]


# --------------------------------------------------------------------------- input helpers


def _number(value: Number, name: str) -> float:
    """A finite real number; strings such as ``"pi/2"`` or ``"1e-3"`` are accepted."""
    if isinstance(value, str):
        return parse_number(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _names(n: int, *, system: bool) -> dict[str, sp.Symbol]:
    """Variable names a formula may use: t and x (one equation) or t, x, y, z, w / x1..x4."""
    names = dict(_core.symbol_table(n))
    if system:
        names.update({f"x{j + 1}": _core.STATE_SYMBOLS[j] for j in range(n)})
    return names


def _expression(value: Expression, names: dict[str, sp.Symbol], what: str) -> sp.Expr:
    """Turn a formula (text, SymPy expression or number) into a SymPy expression."""
    if isinstance(value, str):
        return parse_function(value, names)
    if isinstance(value, sp.Basic):
        replace = {}
        for symbol in value.free_symbols:
            if not isinstance(symbol, sp.Symbol) or symbol.name not in names:
                allowed = ", ".join(names)
                raise ExpressionError(f"{what}: unknown variable {symbol}; allowed: {allowed}")
            replace[symbol] = names[symbol.name]  # also maps Symbol("x", real=True) etc.
        return value.xreplace(replace)
    if isinstance(value, Real) and not isinstance(value, bool):
        return sp.Float(_number(value, what))
    if callable(value):
        raise TypeError(
            f"{what}: give the formula as text (for example 't + x') or as a SymPy expression. "
            "Python functions cannot be used because the strategies need exact symbolic derivatives."
        )
    raise TypeError(f"{what}: expected text, a SymPy expression or a number")


def _check_steps(max_steps: int) -> int:
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    return int(max_steps)


def _numbers(values, name: str) -> np.ndarray:
    """Nested lists / arrays of numbers (or number strings such as ``"0,5"``) -> float array."""
    raw = np.asarray(values, dtype=object)
    flat = [_number(v.item() if isinstance(v, np.generic) else v, name) for v in raw.reshape(-1)]
    return np.asarray(flat, dtype=float).reshape(raw.shape)


def _matrix(A, name: str = "A") -> list[list[float]]:
    rows = _numbers(A, name)
    if rows.ndim != 2 or rows.shape[0] != rows.shape[1] or rows.shape[0] == 0:
        raise ValueError(f"{name} must be a square matrix, got shape {rows.shape}")
    return rows.tolist()


def _vector(values, n: int, name: str = "x0") -> list[float]:
    vec = _numbers(values, name).reshape(-1)
    if vec.size != n:
        raise ValueError(f"{name} must have {n} values to match the {n}x{n} matrix, got {vec.size}")
    return vec.tolist()


# --------------------------------------------------------------------------- one equation


def _solve_scalar(method: str, public_name: str, f: Expression, *, t0, x0, a, b, delta,
                  h_min, max_steps, legacy) -> Solution:
    expr = _expression(f, _names(1, system=False), "f")
    f_fn = sp.lambdify((_T, _X), expr, modules="numpy")
    df_fn = None
    if method in ("error", "pi"):
        total = sp.diff(expr, _X) * expr + sp.diff(expr, _T)  # f_t + f_x * f, exact
        df_fn = sp.lambdify((_T, _X), total, modules="numpy")
    params = {
        "f": str(expr), "t0": _number(t0, "t0"), "x0": _number(x0, "x0"),
        "a": _number(a, "a"), "b": _number(b, "b"), "h_min": _number(h_min, "h_min"),
    }
    if delta is not None:
        params["delta"] = _number(delta, "delta")
    res = _core.solve_scalar(
        method, f_fn, df_fn,
        t0=params["t0"], x0=params["x0"], a=params["a"], b=params["b"], hp=params["h_min"],
        delta=params.get("delta"), max_steps=_check_steps(max_steps), legacy=legacy,
    )
    return Solution(
        method=public_name,
        t=np.asarray(res.t, dtype=float),
        x=np.asarray(res.x, dtype=float),
        h=np.asarray(res.h, dtype=float),
        stopped_by_h_min=res.stopped_by_hp,
        parameters={**params, "legacy": legacy},
    )


def picard(f: Expression, *, x0: Number, a: Number, b: Number, t0: Number = 0.0,
           h_min: Number = 0.0, max_steps: int = DEFAULT_MAX_STEPS,
           legacy: bool = False) -> Solution:
    r"""Step size strategy based on Picard's theorem for ``x' = f(t, x)``.

    Every step is :math:`h_i = \min\{a, b / M_i\}` with
    :math:`M_i = \max |f|` over the rectangle :math:`|t - t_i| \le a,\ |x - x_i| \le b`,
    so the next point stays in the rectangle where Picard's theorem guarantees a
    unique solution. Use it when existence and uniqueness are known.

    Parameters
    ----------
    f:
        Right-hand side, as text (``"t + x"``, ``"x*(1-x)"``, ``"sin(t) - x^2"``), as a
        SymPy expression in ``t`` and ``x``, or as a number.
    x0, t0:
        Initial value ``x(t0) = x0``.
    a:
        Length of the interval ``[t0, t0 + a]``; also the half-width of the rectangle in ``t``.
    b:
        Half-width of the rectangle in ``x``.
    h_min:
        The integration stops if the strategy asks for a step below ``h_min``.
    max_steps:
        Safety limit; :class:`ComputationError` is raised if more steps would be needed.
    legacy:
        ``True`` reproduces the original 2018 web application exactly (``M`` taken on one
        diagonal of the rectangle instead of the whole rectangle).
    """
    return _solve_scalar("picard", "picard", f, t0=t0, x0=x0, a=a, b=b, delta=None,
                         h_min=h_min, max_steps=max_steps, legacy=legacy)


def error_analysis(f: Expression, *, x0: Number, a: Number, b: Number, delta: Number,
                   t0: Number = 0.0, h_min: Number = 0.0,
                   max_steps: int = DEFAULT_MAX_STEPS, legacy: bool = False) -> Solution:
    r"""Step size strategy based on error analysis for ``x' = f(t, x)``.

    Every step is :math:`h_i = \sqrt{2\delta / M_{t_i}}` with
    :math:`M_{t_i} = \max |f_t + f_x f|` over the rectangle around the current point,
    so the local error :math:`\tfrac12 M h^2` of every Euler step stays below ``delta``.
    The derivative :math:`f_t + f_x f` is computed symbolically. Use it when
    existence and uniqueness are known.

    ``delta`` is the desired local error of every step; the other parameters are as
    in :func:`picard`.
    """
    return _solve_scalar("error", "error_analysis", f, t0=t0, x0=x0, a=a, b=b, delta=delta,
                         h_min=h_min, max_steps=max_steps, legacy=legacy)


def picard_error(f: Expression, *, x0: Number, a: Number, b: Number, t0: Number = 0.0,
                 h_min: Number = 0.0, max_steps: int = DEFAULT_MAX_STEPS,
                 legacy: bool = False) -> Solution:
    r"""Step size strategy combining Picard's theorem and error analysis for ``x' = f(t, x)``.

    The rectangle half-width shrinks with an error estimate,
    :math:`b_i = \min\{b_{i-1}, \tfrac12 T_{i-1} h_{i-1}^2\}` with
    :math:`T = \max|f_t + f_x f|`, and the step is :math:`h_i = \min\{a, b_i / M_i\}`.
    Use it when the existence of the solution is not known in advance.

    Parameters are as in :func:`picard`.
    """
    return _solve_scalar("pi", "picard_error", f, t0=t0, x0=x0, a=a, b=b, delta=None,
                         h_min=h_min, max_steps=max_steps, legacy=legacy)


# --------------------------------------------------------------------------- systems


def _system_solution(method: str, res: _core.SystemResult, params: dict, *,
                     with_error: bool) -> Solution:
    return Solution(
        method=method,
        t=np.asarray(res.t, dtype=float),
        x=np.asarray(res.comps, dtype=float).T.copy(),
        h=np.asarray(res.h, dtype=float),
        local_error=np.asarray(res.local_error, dtype=float) if with_error else None,
        stopped_by_h_min=res.stopped_by_hp,
        parameters=params,
    )


def linear_system(A, x0, *, a: Number, b: Number, delta: Number, t0: Number = 0.0,
                  h_min: Number = 0.0, max_steps: int = DEFAULT_MAX_STEPS,
                  legacy: bool = False) -> Solution:
    r"""Step size strategy for linear systems ``X' = A X``, ``X(t0) = x0``.

    Every step is :math:`h_i = \frac{1}{\alpha\sqrt[4]{N^5}}\left(\frac{2\delta}{\beta_{i-1}}\right)^{1/2}`
    with :math:`\alpha = \max|a_{jk}|` and :math:`\beta = \max_j (b + |x_j|)`.
    The local error of every step, :math:`\lVert e^{hA}X_i - X_{i+1}\rVert`, is
    returned in :attr:`Solution.local_error` (matrix exponential via SciPy).

    Parameters
    ----------
    A:
        Constant ``N x N`` coefficient matrix (nested lists or a NumPy array).
    x0:
        Initial vector with ``N`` values.
    a, b, delta, t0, h_min, max_steps:
        As in :func:`error_analysis`; ``delta`` is the desired local error level.
    legacy:
        ``True`` reproduces the original web application, which used the element-wise
        exponential instead of the matrix exponential for the local error.
    """
    matrix = _matrix(A)
    n = len(matrix)
    start = _vector(x0, n)
    params = {"A": matrix, "x0": start, "t0": _number(t0, "t0"), "a": _number(a, "a"),
              "b": _number(b, "b"), "delta": _number(delta, "delta"),
              "h_min": _number(h_min, "h_min"), "legacy": legacy}
    res = _core.solve_linear_system(
        matrix, start, t0=params["t0"], a=params["a"], b=params["b"], hp=params["h_min"],
        delta=params["delta"], max_steps=_check_steps(max_steps), legacy=legacy,
    )
    return _system_solution("linear_system", res, params, with_error=True)


def nonlinear_system(A, phi: Sequence[Expression], x0, *, a: Number, b: Number,
                     delta: Number, t0: Number = 0.0, h_min: Number = 0.0,
                     max_steps: int = DEFAULT_MAX_STEPS, legacy: bool = False) -> Solution:
    r"""Step size strategy for non-linear systems ``X' = A X + phi(t, X)``, ``X(t0) = x0``.

    Every step is
    :math:`h_i = N^{-1/4}\left(\frac{2\delta}{N^2\alpha^2\beta_{i-1} + N\alpha\gamma_{i-1} + \zeta_{i-1}}\right)^{1/2}`,
    where :math:`\gamma` bounds :math:`|\varphi|` and :math:`\zeta` bounds
    :math:`|d\varphi/dt| = |\varphi_t + J_x(\varphi)(AX + \varphi)|`. The total derivative
    is computed symbolically with SymPy.

    Parameters
    ----------
    A:
        Constant ``N x N`` matrix of the linear part, ``N <= 4``.
    phi:
        ``N`` formulas for the non-linear part. Variables: ``t`` and ``x, y, z, w``
        (or ``x1, x2, x3, x4``), for example ``["-0.5*x*y", "0.25*x*y"]``.
    x0:
        Initial vector with ``N`` values.
    a, b, delta, t0, h_min, max_steps:
        As in :func:`linear_system`.
    legacy:
        ``True`` reproduces the original web application, which combined the terms of
        the derivative of ``phi`` in the wrong order.
    """
    matrix = _matrix(A)
    n = len(matrix)
    if n > len(_core.STATE_SYMBOL_NAMES):
        raise ValueError(f"nonlinear_system supports up to {len(_core.STATE_SYMBOL_NAMES)} equations")
    if isinstance(phi, (str, sp.Basic)):
        raise ValueError(f"phi must be a list of {n} formulas, one per equation")
    phi = list(phi)
    if len(phi) != n:
        raise ValueError(f"phi must be a list of {n} formulas, one per equation, got {len(phi)}")
    start = _vector(x0, n)
    names = _names(n, system=True)
    exprs = [_expression(p, names, f"phi[{j}]") for j, p in enumerate(phi)]
    model = _core.build_nonlinear_model(matrix, exprs, legacy=legacy)
    params = {"A": matrix, "phi": [str(e) for e in exprs], "x0": start, "t0": _number(t0, "t0"),
              "a": _number(a, "a"), "b": _number(b, "b"), "delta": _number(delta, "delta"),
              "h_min": _number(h_min, "h_min"), "legacy": legacy}
    res = _core.solve_nonlinear_system(
        matrix, start, model, t0=params["t0"], a=params["a"], b=params["b"],
        hp=params["h_min"], delta=params["delta"], max_steps=_check_steps(max_steps),
    )
    return _system_solution("nonlinear_system", res, params, with_error=False)


__all__ = [
    "ComputationError", "ExpressionError", "picard", "error_analysis", "picard_error",
    "linear_system", "nonlinear_system",
]
