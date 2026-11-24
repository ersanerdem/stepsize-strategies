# stepsize-strategies

Variable step size strategies for the Euler method, as a Python package.

The step of every Euler step is chosen **before** the step is taken, from a bound that
guarantees either that the next point stays in the rectangle of Picard's theorem or that the
local error stays below a tolerance. The five strategies come from the work of the research
group at Necmettin Erbakan University and are the numerical core of the
[Step Size Strategies web application](https://stepsizestrategies.pythonanywhere.com).

| Function | Problem | Step size |
| --- | --- | --- |
| `picard` | `x' = f(t, x)` | `h_i = min{a, b / M_i}`,  `M_i = max \|f\|` |
| `error_analysis` | `x' = f(t, x)` | `h_i = sqrt(2 δ / M_ti)`,  `M_ti = max \|f_t + f_x f\|` |
| `picard_error` | `x' = f(t, x)` | `h_i = min{a, b_i / M_i}`, the rectangle `b_i` shrinks with an error estimate |
| `linear_system` | `X' = A X` | `h_i = (2 δ / β_{i-1})^(1/2) / (α N^(5/4))` |
| `nonlinear_system` | `X' = A X + φ(t, X)` | `h_i = N^(-1/4) (2 δ / (N² α² β + N α γ + ζ))^(1/2)` |

`M`, `β`, `γ`, `ζ` are bounds of the function and its derivatives around the current point;
`α = max |a_jk|`. All derivatives (`f_t + f_x f`, and `φ_t + J_x(φ)(AX + φ)` for non-linear
systems) are computed **exactly with SymPy**, not with finite differences.

## Installation

```bash
pip install stepsize-strategies            # numpy, scipy, sympy
pip install "stepsize-strategies[all]"     # + pandas and matplotlib for tables and plots
```

Python 3.10 or newer.

## Quick start

```python
import stepsize_strategies as sss

# One equation: x' = t + x, x(0) = 1 on [0, 1]
sol = sss.error_analysis("t + x", x0=1, a=1, b=1, delta=1e-3)

sol                 # Solution(method='error_analysis', steps=..., t_final=1, reached t0 + a)
sol.t, sol.x        # time points and the solution
sol.h               # the step sizes chosen by the strategy
sol.summary()       # steps, final value, smallest / largest step ...
sol.to_dataframe()  # the table of steps (pandas)
sol.plot()          # solution and step sizes (matplotlib)
```

```python
# Linear system X' = A X: a damped oscillator
sol = sss.linear_system([[0, 1], [-1, -0.3]], x0=[1, 0], a=5, b=1, delta=5e-3)
sol.x.shape         # (steps + 1, 2)
sol.local_error     # local error of every step, || e^{hA} X_i - X_{i+1} ||

# Non-linear system X' = A X + φ(t, X): a predator-prey model
sol = sss.nonlinear_system(
    A=[[1, 0], [0, -0.5]],
    phi=["-0.5*x*y", "0.25*x*y"],
    x0=[2, 1], a=5, b=1, delta=5e-3,
)
```

## Parameters

| Name | Meaning |
| --- | --- |
| `f`, `phi` | Formulas as text (`"x*(1-x)"`, `"sin(t) - x^2"`, `"exp(-t)*y"`) or SymPy expressions. Variables: `t`, `x` for one equation; `t`, `x, y, z, w` (or `x1 … x4`) for systems. Functions: `sin cos tan cot sec csc asin acos atan sinh cosh tanh exp log ln sqrt abs`, constants `pi`, `E`. |
| `x0`, `t0` | Initial value `x(t0) = x0` (`t0` defaults to 0). A list for systems. |
| `A` | Constant `N x N` matrix (lists or a NumPy array). `nonlinear_system` supports `N <= 4`. |
| `a` | Length of the interval `[t0, t0 + a]` (and the half-width of the rectangle in `t`). |
| `b` | Half-width of the rectangle around the solution. |
| `delta` | Desired local error of every step (`error_analysis` and the systems). |
| `h_min` | The integration stops when the strategy asks for a smaller step; `sol.stopped_by_h_min` is then `True`. Default 0. |
| `max_steps` | Safety limit (default 100 000); `ComputationError` if more steps are needed. |
| `legacy` | `True` reproduces the numbers of the original 2018 web application (see below). |

Numbers may also be given as text: `"pi/4"`, `"1e-3"`, `"0,5"`. Formulas are parsed by a
safe parser (no `eval`), so text typed by users can be passed in directly.

## Which strategy?

- Existence and uniqueness of the solution known: `picard` or `error_analysis`.
- Existence not known in advance: `picard_error`.
- Systems: `linear_system` for `X' = A X`, `nonlinear_system` for `X' = A X + φ(t, X)`.

## Legacy mode

By default three details of the original 2018 web application are corrected: the bound `M`
is taken over the whole rectangle (the original looked at one diagonal), the local error of
linear systems uses the matrix exponential (the original used the element-wise exponential),
and the terms of `dφ/dt` are combined in the right order. `legacy=True` gives the original
numbers, for example to reproduce published tables.

## Try it in the browser

The same strategies, with ready-made examples, charts and Excel export:
<https://stepsizestrategies.pythonanywhere.com> (English) ·
<https://stepsizestrategies.pythonanywhere.com/tr/> (Türkçe).

## How to cite

If you use this package, please cite:

> E. Erdem, G. Çelik Kızılkan, A. O. Çıbıkdiken. *Web Application for Step Size Strategies.*
> Konuralp Journal of Mathematics 7(2), 475–485, 2019.
> [DergiPark](https://dergipark.org.tr/en/pub/konuralpjournalmath/article/618922)

GitHub's "Cite this repository" button gives the same reference (`CITATION.cff`).

The package grew out of the master's thesis *Adım Genişliği Stratejileri İçin Web Uygulaması*
(Web Application for Step Size Strategies), E. Erdem, Necmettin Erbakan University, 2019.

## The strategies in the literature

- A new variable step size algorithm for Cauchy problem.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0096300306006217)
- Step Size Strategies Based on Error Analysis.
  [DergiPark](http://dergipark.gov.tr/download/article-file/214950)
- Step Size Strategy for the Numerical Integration of Systems of the Linear Systems.
  [ResearchGate](https://www.researchgate.net/publication/260391200_Step_Size_Strategies_Based_On_Error_Analysis_For_The_Linear_Systems)
- Step Size Strategy for the Numerical Integration of Systems of Differential Equations.
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0377042711003682)

## Türkçe

Euler yöntemi için değişken adım uzunluğu stratejileri: Picard teoremi, hata analizi,
Picard + hata analizi, lineer ve lineer olmayan sistemler. Kurulum `pip install
stepsize-strategies`; kullanım örnekleri yukarıda. Aynı stratejiler tarayıcıda:
<https://stepsizestrategies.pythonanywhere.com/tr/>.

## License

MIT © 2026 Ersan Erdem
