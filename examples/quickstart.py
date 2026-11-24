"""Quick tour of stepsize-strategies. Run:  python examples/quickstart.py"""

import stepsize_strategies as sss

print("stepsize-strategies", sss.__version__)

# One equation: x' = t + x, x(0) = 1 on [0, 1]; exact solution x(1) = 2e - 2 = 3.43656...
for solver, extra in [(sss.picard, {}), (sss.error_analysis, {"delta": 1e-4})]:
    sol = solver("t + x", x0=1, a=1, b=1, **extra)
    print(f"{sol.method:15s} steps={sol.steps:5d}  x(1)={sol.x[-1]:.5f}")

# Existence not known in advance: Picard + error analysis (stops when the step gets too small)
sol = sss.picard_error("x^2", x0=1, a=1, b=1, h_min=1e-4)
print(sol, sol.summary()["largest_step"])

# Linear system: damped oscillator
sol = sss.linear_system([[0, 1], [-1, -0.3]], x0=[1, 0], a=5, b=1, delta=5e-3)
print(sol, "largest local error:", sol.summary()["largest_local_error"])

# Non-linear system: predator-prey
sol = sss.nonlinear_system([[1, 0], [0, -0.5]], ["-0.5*x*y", "0.25*x*y"], x0=[2, 1],
                           a=5, b=1, delta=5e-3)
print(sol, "X(5) =", sol.x[-1])

try:
    print(sol.to_dataframe().head())
    sol.plot()
    import matplotlib.pyplot as plt
    plt.show()
except ImportError as exc:
    print(exc)
