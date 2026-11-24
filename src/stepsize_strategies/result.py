"""The object returned by every solver."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Solution:
    """Numerical solution produced by one step size strategy.

    Attributes
    ----------
    method:
        Name of the strategy (``"picard"``, ``"error_analysis"``, ``"picard_error"``,
        ``"linear_system"`` or ``"nonlinear_system"``).
    t:
        Time points ``t_0 ... t_n`` (``n + 1`` values).
    x:
        Solution at ``t``. Shape ``(n + 1,)`` for one equation, ``(n + 1, N)`` for a
        system of ``N`` equations.
    h:
        Step sizes ``h_0 ... h_{n-1}`` (``n`` values).
    local_error:
        Local error of every step (linear systems only), otherwise ``None``.
    stopped_by_h_min:
        ``True`` when the strategy asked for a step smaller than ``h_min`` before the
        end of the interval ``[t0, t0 + a]`` was reached.
    parameters:
        The inputs of the run, for reference.
    """

    method: str
    t: np.ndarray
    x: np.ndarray
    h: np.ndarray
    local_error: np.ndarray | None = None
    stopped_by_h_min: bool = False
    parameters: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic facts

    @property
    def steps(self) -> int:
        """Number of Euler steps taken."""
        return int(self.h.size)

    @property
    def dimension(self) -> int:
        """1 for a single equation, N for a system of N equations."""
        return 1 if self.x.ndim == 1 else int(self.x.shape[1])

    @property
    def norm(self) -> np.ndarray:
        """Euclidean norm of the solution at every time point (``|x|`` for one equation)."""
        return np.abs(self.x) if self.x.ndim == 1 else np.linalg.norm(self.x, axis=1)

    @property
    def reached_end(self) -> bool:
        """``True`` when the integration covered the whole interval ``[t0, t0 + a]``."""
        return not self.stopped_by_h_min

    # ------------------------------------------------------------------ summaries

    def summary(self) -> dict:
        """Key numbers of the run as a plain dictionary."""
        final = float(self.x[-1]) if self.x.ndim == 1 else [float(v) for v in self.x[-1]]
        info = {
            "method": self.method,
            "steps": self.steps,
            "t_final": float(self.t[-1]),
            "x_final": final,
            "smallest_step": float(self.h.min()) if self.steps else None,
            "largest_step": float(self.h.max()) if self.steps else None,
            "reached_end": self.reached_end,
        }
        if self.local_error is not None and self.local_error.size:
            info["largest_local_error"] = float(self.local_error.max())
        return info

    def to_dataframe(self):
        """Table of the steps as a :class:`pandas.DataFrame` (needs ``pandas``).

        One row per step: the step number, ``t_i``, ``h_i``, the state at ``t_i``,
        its norm (systems) and the local error (linear systems).
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_dataframe() needs pandas: pip install stepsize-strategies[pandas]"
            ) from exc
        n = self.steps
        data = {"step": np.arange(n), "t": self.t[:n], "h": self.h}
        if self.x.ndim == 1:
            data["x"] = self.x[:n]
        else:
            for j in range(self.dimension):
                data[f"x{j + 1}"] = self.x[:n, j]
            data["norm"] = self.norm[:n]
        if self.local_error is not None:
            data["local_error"] = self.local_error
        return pd.DataFrame(data)

    def plot(self, ax=None, *, steps: bool = True):
        """Plot the solution (and, with ``steps=True``, the step sizes) with matplotlib.

        Returns the axes used. Needs ``matplotlib``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "plot() needs matplotlib: pip install stepsize-strategies[plot]"
            ) from exc
        if ax is None:
            if steps:
                _, (ax, ax_h) = plt.subplots(2, 1, sharex=True, figsize=(7, 5.5),
                                             gridspec_kw={"height_ratios": [3, 1.4]})
            else:
                _, ax = plt.subplots(figsize=(7, 4))
                ax_h = None
        else:
            ax_h = None
        if self.x.ndim == 1:
            ax.plot(self.t, self.x, marker=".", label="x(t)")
        else:
            for j in range(self.dimension):
                ax.plot(self.t, self.x[:, j], marker=".", label=f"x{j + 1}(t)")
        ax.set_ylabel("solution")
        ax.set_title(f"{self.method}: {self.steps} steps")
        ax.legend()
        ax.grid(alpha=0.3)
        if ax_h is not None and self.steps:
            ax_h.step(self.t[:-1], self.h, where="post")
            ax_h.set_ylabel("step size h")
            ax_h.set_xlabel("t")
            ax_h.grid(alpha=0.3)
        else:
            ax.set_xlabel("t")
        return ax

    def __repr__(self) -> str:
        end = "reached t0 + a" if self.reached_end else "stopped: step below h_min"
        return (
            f"Solution(method={self.method!r}, steps={self.steps}, "
            f"t_final={float(self.t[-1]):.6g}, {end})"
        )
