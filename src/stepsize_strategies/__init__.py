"""Variable step size strategies for the Euler method.

Five strategies from the work of the research group at Necmettin Erbakan University
(Çelik Kızılkan; Çelik Kızılkan and Aydın) for

* one equation ``x' = f(t, x)``: :func:`picard`, :func:`error_analysis`, :func:`picard_error`
* linear systems ``X' = A X``: :func:`linear_system`
* non-linear systems ``X' = A X + phi(t, X)``: :func:`nonlinear_system`

Example
-------
>>> import stepsize_strategies as sss
>>> sol = sss.error_analysis("t + x", x0=1, a=1, b=1, delta=1e-3)
>>> sol.reached_end
True
"""

from ._core import ComputationError
from ._expr import ExpressionError
from .api import error_analysis, linear_system, nonlinear_system, picard, picard_error
from .result import Solution

__version__ = "0.1.0"

__all__ = [
    "ComputationError",
    "ExpressionError",
    "Solution",
    "error_analysis",
    "linear_system",
    "nonlinear_system",
    "picard",
    "picard_error",
    "__version__",
]
