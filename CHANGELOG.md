# Changelog

## 0.1.0 (2026-11-24)

First release. The numerical core of the
[Step Size Strategies web application](https://stepsizestrategies.pythonanywhere.com)
as a Python package:

- `picard`, `error_analysis`, `picard_error` for one equation `x' = f(t, x)`
- `linear_system` for `X' = A X` (local error with the matrix exponential)
- `nonlinear_system` for `X' = A X + phi(t, X)`, up to 4 equations
- exact symbolic derivatives with SymPy, safe parsing of formulas given as text
- `Solution` object with `summary()`, `to_dataframe()` and `plot()`
- `legacy=True` reproduces the numbers of the original 2018 web application
